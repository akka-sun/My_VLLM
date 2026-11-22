import hashlib
import json
from pathlib import Path
import statistics

from transformers import AutoTokenizer
from vllm.benchmarks.datasets import ShareGPTDataset

root = Path('/root/autodl-tmp')
dataset = root / 'datasets/sharegpt'
tokenizer = AutoTokenizer.from_pretrained(root / 'models/Qwen--Qwen3-0.6B', local_files_only=True)
requests = ShareGPTDataset(dataset_path=str(dataset / 'ShareGPT_V3_unfiltered_cleaned_split.json'),
                           random_seed=42).sample(tokenizer, num_requests=256, output_len=128)
lengths = [sample.prompt_len for sample in requests]
report = {
    'source': json.loads((dataset / 'manifest.json').read_text()),
    'seed': 42, 'samples': len(requests), 'output_tokens_per_request': 128,
    'prompt_lengths': lengths,
    'prompt_tokens_total': sum(lengths),
    'prompt_tokens_min': min(lengths), 'prompt_tokens_median': statistics.median(lengths),
    'prompt_tokens_max': max(lengths),
    'ordered_sample_sha256': hashlib.sha256(
        json.dumps([(sample.prompt, sample.expected_output_len) for sample in requests],
                   ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
}
(root / 'runs/vllm-sharegpt-pressure/sample.json').write_text(json.dumps(report, indent=2) + '\n')
print({k: v for k, v in report.items() if k not in ('prompt_lengths','source')})
