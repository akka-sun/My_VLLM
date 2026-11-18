"""Measure streaming chat completions against a running HTTP server."""
import argparse
import asyncio
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import httpx


def parse_sse(lines):
    texts, token_count, done = [], None, False
    data = []
    for line in lines:
        line = line.rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif not line and data:
            payload = '\n'.join(data)
            data.clear()
            if payload == '[DONE]':
                done = True
                continue
            event = json.loads(payload)
            if event.get('error'):
                raise ValueError(str(event['error']))
            if event.get('usage') is not None:
                token_count = event['usage'].get('completion_tokens')
            for choice in event.get('choices', []):
                content = choice.get('delta', {}).get('content')
                if content:
                    texts.append(content)
    return texts, token_count, done


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


async def run_one(client, url, item, model, max_tokens, timeout):
    body = {'model': model, 'messages': item['messages'], 'max_tokens': max_tokens,
            'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
    result = {'id': item['id'], 'ttft_ms': None, 'e2e_ms': None, 'completion_tokens': None, 'error': None}
    start = time.perf_counter()
    lines = []
    texts_seen, count, done = False, None, False
    try:
        async with client.stream('POST', url, json=body, timeout=timeout) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                lines.append(line)
                if not line:
                    parts, event_count, event_done = parse_sse(lines)
                    lines.clear()
                    if event_count is not None:
                        count = event_count
                    done = done or event_done
                    if parts:
                        texts_seen = True
                        if result['ttft_ms'] is None:
                            result['ttft_ms'] = (time.perf_counter() - start) * 1000
        parts, event_count, event_done = parse_sse(lines + [''])
        texts_seen = texts_seen or bool(parts)
        count = event_count if event_count is not None else count
        done = done or event_done
        if not done or not texts_seen:
            raise ValueError('stream ended without [DONE] or text')
        result['completion_tokens'] = count
    except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result['error'] = str(exc)
    result['e2e_ms'] = (time.perf_counter() - start) * 1000
    return result


async def run(args):
    raw = Path(args.workload).read_bytes()
    items = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    if not items or args.concurrency < 1 or args.max_tokens < 1:
        raise ValueError('workload, concurrency and max_tokens must be positive')
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        async def bounded(item):
            async with semaphore:
                return await run_one(client, args.url, item, args.model, args.max_tokens, args.timeout)
        start = time.perf_counter()
        results = await asyncio.gather(*(bounded(item) for item in items))
        elapsed = time.perf_counter() - start
    successes = [r for r in results if r['error'] is None]
    counts_available = all(r['completion_tokens'] is not None for r in successes) and bool(successes)
    metrics = {'requests': len(results), 'successes': len(successes), 'failures': len(results)-len(successes),
               'elapsed_seconds': elapsed, 'requests_per_second': len(successes)/elapsed,
               'output_tokens_per_second': sum(r['completion_tokens'] for r in successes)/elapsed if counts_available else None,
               'ttft_p95_ms': percentile([r['ttft_ms'] for r in successes if r['ttft_ms'] is not None], .95),
               'e2e_p95_ms': percentile([r['e2e_ms'] for r in successes], .95)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'requests.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in results), encoding='utf-8')
    report = {'workload_sha256': hashlib.sha256(raw).hexdigest(),
              'configuration': {'model': args.model, 'concurrency': args.concurrency, 'max_tokens': args.max_tokens},
              'environment': {'platform': platform.platform(), 'python': platform.python_version(), 'url': args.url},
              'metrics': metrics}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='full /v1/chat/completions URL')
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', required=True, help='JSONL with id and messages; messages may include image_url parts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=120)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
