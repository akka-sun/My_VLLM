import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kv', choices=['auto', 'fp8_e4m3'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    path = '/root/autodl-tmp/models/Qwen--Qwen3-0.6B'
    options = dict(dtype='bfloat16', kv_cache_dtype=args.kv,
                   calculate_kv_scales=False, enforce_eager=True,
                   max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=512,
                   block_size=32, num_gpu_blocks_override=1536, gpu_memory_utilization=.35,
                   enable_prefix_caching=False, disable_log_stats=True, seed=42)
    engine = LLM(model=path, **options)
    vocab = len(engine.get_tokenizer())
    device = torch.cuda.get_device_properties(0)
    import os
    summary = dict(vllm=vllm.__version__, torch=torch.__version__, gpu=device.name,
                   model=path, revision='c1899de289a04d12100db370d81485cdf75e47ca',
                   backend=os.environ.get('VLLM_ATTENTION_BACKEND'), options=options, scenarios={})
    for name, lengths, out in [
        ('mixed', [2048 if i % 5 == 0 else 128 for i in range(20)], 64),
        ('decode-heavy', [512] * 20, 256),
    ]:
        params = SamplingParams(temperature=0, max_tokens=out, ignore_eos=True, detokenize=False)
        rows = []
        for rep in range(-2, 5):
            rng = random.Random(42 + rep)
            ids = [[rng.randrange(vocab) for _ in range(size)] for size in lengths]
            prompts = [TokensPrompt(prompt_token_ids=p) for p in ids]
            start = time.perf_counter()
            results = engine.generate(prompts, params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            count = sum(len(r.outputs[0].token_ids) for r in results)
            assert count == len(lengths) * out
            rec = dict(repeat=rep, output_tokens=count, seconds=elapsed, tokens_per_second=count/elapsed,
                       workload_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            print(name, json.dumps(rec), flush=True)
            if rep >= 0:
                rows.append(rec)
        summary['scenarios'][name] = dict(input_lengths=lengths, output_tokens_per_request=out,
            runs=rows, median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in rows))
    args.out.write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    main()
