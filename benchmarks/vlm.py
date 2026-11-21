"""Measure local Qwen3-VL image-to-text generation on one selected GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompt', default='Describe this image.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.repeats) < 1:
        parser.error('token and repeat counts must be positive')

    import torch
    from nanovllm.models.qwen3_vl import Qwen3VLAdapter

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('a CUDA device is required for this benchmark')
    adapter = Qwen3VLAdapter(args.model, args.device)
    inputs = adapter.prepare_image(args.image, args.prompt)
    adapter.generate(inputs, min(args.max_new_tokens, 8))
    runs = []
    for _ in range(args.repeats):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = adapter.generate(inputs, args.max_new_tokens)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        runs.append(dict(output, seconds=elapsed, output_tokens_per_second=output['output_tokens'] / elapsed,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
    report = {'model': str(args.model.resolve()), 'gpu': torch.cuda.get_device_name(device),
              'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
              'prompt': args.prompt, 'input_tokens': inputs.input_ids.shape[1],
              'image_grid_thw': inputs.image_grid_thw.tolist(), 'runs': runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
