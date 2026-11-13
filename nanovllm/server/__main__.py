import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Serve Nano-vLLM with an OpenAI-compatible API')
    parser.add_argument('--model', required=True, help='Local floating-point safetensors model directory')
    parser.add_argument('--served-model-name', default='nano-vllm')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-num-batched-tokens', type=int, default=512)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--scheduling-policy', choices=['fcfs', 'priority'], default='fcfs')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=0)
    parser.add_argument('--long-prefill-token-threshold-adaptive', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--scheduler-reserve-full-isl', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--max-pending-requests', type=int, default=256)
    parser.add_argument('--quantization', choices=['int8'], default=None)
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--speculative-config', type=json.loads, default=None,
                        help='JSON: method (draft_model/dspark), local model, num_speculative_tokens')
    args = parser.parse_args()
    try:
        import uvicorn
        from nanovllm.server.app import create_app
    except ImportError as exc:
        parser.error(f'Install server dependencies with pip install ".[server]": {exc}')
    options = {name:getattr(args, name) for name in (
        'tensor_parallel_size', 'max_model_len', 'max_num_batched_tokens', 'max_num_seqs',
        'gpu_memory_utilization', 'quantization', 'enforce_eager', 'scheduling_policy',
        'long_prefill_token_threshold', 'long_prefill_token_threshold_adaptive',
        'scheduler_reserve_full_isl', 'speculative_config')}
    app = create_app(args.model, served_model_name=args.served_model_name,
        engine_kwargs=options, max_pending_requests=args.max_pending_requests)
    # One HTTP worker owns the engine and its tensor-parallel subprocesses.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
