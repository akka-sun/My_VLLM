import json
from pathlib import Path

root = Path(__file__).resolve().parent
cases = [
    ('normal-auto', 'BF16 / 49,152'),
    ('normal-fp8_e4m3', 'FP8 / 49,152'),
    ('small-auto', 'BF16 / 12,288'),
    ('small-fp8_e4m3', 'FP8 / 12,288'),
    ('equal-bytes-fp8_e4m3', 'FP8 / 24,576'),
]
sample = json.loads((root / 'sample.json').read_text())

def preemptions(name, when):
    matches = [line for line in (root / f'{name}.{when}.txt').read_text().splitlines()
               if line.startswith('vllm:num_preemptions_total{')]
    assert matches, (name, when)
    return sum(int(float(line.rsplit(' ', 1)[1])) for line in matches)

print('| KV / 容量 token | 并发 | 输出 token/s | P95 TTFT ms | P95 TPOT ms | P95 E2E ms | 抢占 | 失败 |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for c in [32, 64, 128]:
    for tag, label in cases:
        name = f'{tag}-c{c}'
        path = root / f'{name}.json'
        if not path.is_file():
            continue
        m = json.loads(path.read_text())
        assert m['num_prompts'] == sample['samples'] and m['max_concurrency'] == c
        failures = m['num_prompts'] - m['completed']
        if not failures:
            assert m['total_input_tokens'] == sample['prompt_tokens_total'], name
            assert m['total_output_tokens'] == sample['output_tokens_per_request'] * m['completed'], name
        count = preemptions(name, 'after') - preemptions(name, 'before')
        print(f"| {label} | {c} | {m['output_throughput']:.2f} | {m['p95_ttft_ms']:.1f} | "
              f"{m['p95_tpot_ms']:.1f} | {m['p95_e2el_ms']:.1f} | {count} | {failures} |")
