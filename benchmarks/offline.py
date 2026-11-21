"""Run a reproducible offline Nano-vLLM benchmark on one or more GPUs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', help='CUDA_VISIBLE_DEVICES selection, e.g. 0,1')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--mode', choices=['baseline', 'graph', 'chunked', 'kv-int8-reference'], default='baseline')
    parser.add_argument('--kv-int8-layers', type=int, nargs='+', help='Zero-based INT8 layers; default: all')
    parser.add_argument('--kv-cache-max-blocks', type=int, help='Cap pages for equal-capacity comparisons')
    parser.add_argument('--requests', type=int, default=80)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--short-input', type=int, default=128)
    parser.add_argument('--long-input', type=int, default=2048)
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-seqs', type=int, default=32)
    parser.add_argument('--token-budget', type=int, default=512)
    parser.add_argument('--prefill-threshold', type=int, default=128)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.35)
    args = parser.parse_args()
    if not 1 <= args.tp <= 8 or min(args.requests, args.repeats, args.short_input,
                                    args.long_input, args.output_tokens) < 1 or args.warmup < 0:
        parser.error('TP must be in [1, 8], lengths and counts must be positive, and warmup cannot be negative')
    if args.kv_int8_layers is not None and args.mode != 'kv-int8-reference':
        parser.error('--kv-int8-layers requires --mode kv-int8-reference')
    if args.devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    import torch
    from transformers import AutoConfig
    from nanovllm import LLM, SamplingParams

    if torch.cuda.device_count() < args.tp:
        parser.error(f'TP={args.tp} requires {args.tp} visible GPUs, found {torch.cuda.device_count()}')
    model = args.model.resolve()
    hf = AutoConfig.from_pretrained(model, local_files_only=True)
    if hf.model_type != 'qwen3':
        parser.error('the offline Nano-vLLM benchmark currently supports text Qwen3')
    if hf.num_attention_heads % args.tp or hf.num_key_value_heads % args.tp:
        parser.error('TP must divide both the attention head count and the KV head count')
    if max(args.short_input, args.long_input) + args.output_tokens > min(args.max_model_len, hf.max_position_embeddings):
        parser.error('input plus output exceeds the configured model context')
    options = dict(tensor_parallel_size=args.tp, enforce_eager=args.mode != 'graph',
                   kv_cache_dtype='int8-reference' if args.mode == 'kv-int8-reference' else 'auto',
                   kv_int8_layers=args.kv_int8_layers, kv_cache_max_blocks=args.kv_cache_max_blocks,
                   long_prefill_token_threshold=args.prefill_threshold if args.mode == 'chunked' else 0,
                   max_num_seqs=args.max_seqs, max_num_batched_tokens=args.token_budget,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    gpus = []
    for rank in range(args.tp):
        props = torch.cuda.get_device_properties(rank)
        gpus.append({'logical_device': rank, 'name': props.name, 'uuid': str(props.uuid),
                     'total_memory_bytes': props.total_memory})
    preparation = model / 'preparation.json'
    revision = json.loads(preparation.read_text(encoding='utf-8')).get('revision') if preparation.exists() else None
    manifest = {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                           cwd=Path(__file__).resolve().parents[1]).strip(),
        'gpu': gpus, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'torchdynamo_disable': os.environ.get('TORCHDYNAMO_DISABLE'),
        'model': str(model), 'model_revision': revision, 'mode': args.mode, 'options': options,
        'seed': args.seed, 'requests': args.requests, 'warmup': args.warmup,
        'repeats': args.repeats, 'input_lengths': [args.short_input, args.long_input],
        'output_tokens': args.output_tokens,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.output_tokens)

    def prompts(seed, count):
        rng = random.Random(seed)
        return [[rng.randrange(vocabulary_size) for _ in range(args.long_input if i % 5 == 0 else args.short_input)]
                for i in range(count)]

    rows = []
    print(f'Loading {model.name} on {args.tp} GPU(s), mode={args.mode}', file=sys.stderr, flush=True)
    engine = LLM(str(model), **options)
    try:
        vocabulary_size = min(hf.vocab_size, len(engine.tokenizer))
        manifest['sampled_vocabulary_size'] = vocabulary_size
        blocks = engine.config.num_kvcache_blocks
        bytes_per_block = engine.model_runner.kv_bytes_per_block
        manifest['rank0_kv_pool'] = {'blocks': blocks, 'bytes_per_block': bytes_per_block,
                                     'allocated_bytes': blocks * bytes_per_block,
                                     'reference_workspace_budget_bytes': engine.model_runner.kv_workspace_budget_bytes}
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if args.warmup:
            print(f'Warming up with {args.warmup} requests', file=sys.stderr, flush=True)
            engine.generate(prompts(args.seed - 1, args.warmup), params, use_tqdm=False)
        for repeat in range(args.repeats):
            print(f'Measuring run {repeat + 1}/{args.repeats}', file=sys.stderr, flush=True)
            workload = prompts(args.seed + repeat, args.requests)
            digest = hashlib.sha256(json.dumps(workload).encode()).hexdigest()
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            outputs = engine.generate(workload, params, use_tqdm=False)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - start
            tokens = sum(len(output['token_ids']) for output in outputs)
            if len(outputs) != args.requests or tokens != args.requests * args.output_tokens:
                raise RuntimeError('engine returned fewer requests or tokens than the fixed workload requires')
            row = {'repeat': repeat, 'workload_sha256': digest, 'seconds': elapsed,
                   'output_tokens': tokens, 'output_tokens_per_second': tokens / elapsed,
                   'rank0_peak_allocated_bytes_including_kv_pool': torch.cuda.max_memory_allocated(0)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        engine.exit()
    throughput = statistics.median(row['output_tokens_per_second'] for row in rows)
    summary = {'configuration': manifest, 'runs': rows,
               'median_output_tokens_per_second': throughput,
               'median_output_tokens_per_second_per_gpu': throughput / args.tp}
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
