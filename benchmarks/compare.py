"""Compare benchmark summaries for the same workload."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    if baseline.get('workload_sha256') != candidate.get('workload_sha256') or not baseline.get('workload_sha256'):
        raise ValueError('workload hashes must match')
    result = {}
    for key, direction in [('output_tokens_per_second', 1), ('ttft_p95_ms', -1), ('e2e_p95_ms', -1)]:
        a = baseline['metrics'].get(key)
        b = candidate['metrics'].get(key)
        result[key] = None if a is None or b is None or a <= 0 else direction * (b / a - 1) * 100
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    args = parser.parse_args()
    a = json.loads(Path(args.baseline).read_text(encoding='utf-8'))
    b = json.loads(Path(args.candidate).read_text(encoding='utf-8'))
    if a.get('configuration') != b.get('configuration'):
        raise ValueError('model, load, or generation settings differ; comparisons are not equivalent')
    print(json.dumps({'percentage_improvement': compare(a, b), 'baseline': a['metrics'], 'candidate': b['metrics']}, indent=2))


if __name__ == '__main__':
    main()
