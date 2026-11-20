"""Generate reproducible text requests; lengths are text repetitions, not tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def make_workload(preset, count, seed):
    if preset not in {'short', 'long', 'mixed'} or count < 1:
        raise ValueError('preset must be short/long/mixed and count must be positive')
    rng = random.Random(seed)
    scenarios = [preset] * count
    if preset == 'mixed':
        long_count = max(1, count // 5)
        scenarios = ['long'] * long_count + ['short'] * (count - long_count)
        rng.shuffle(scenarios)
    rows = []
    for i, scenario in enumerate(scenarios):
        topic = rng.choice(['caching', 'scheduling', 'parallel computing', 'memory management'])
        context = f'This document discusses {topic}. Keep the explanation clear and precise. '
        prompt = context * (128 if scenario == 'long' else 4)
        prompt += f'\nRequest {i}: summarize the document in three sentences.'
        rows.append({'id': str(i), 'scenario': scenario,
                     'messages': [{'role': 'user', 'content': prompt}]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['short', 'long', 'mixed'], default='mixed')
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n'
                   for row in make_workload(args.preset, args.count, args.seed)).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(data).hexdigest(),
                      'count': args.count, 'seed': args.seed, 'preset': args.preset}))


if __name__ == '__main__':
    main()
