import json
import pytest
from benchmarks.compare import compare
from benchmarks.serve import parse_sse


def test_stream_ignores_role_and_counts_usage():
    events = ['data: {"choices":[{"delta":{"role":"assistant"}}]}', '',
              'data: {"choices":[{"delta":{"content":"hello"}}]}', '',
              'data: {"choices":[],"usage":{"completion_tokens":2}}', '', 'data: [DONE]', '']
    result = parse_sse(events)
    assert result == (['hello'], 2, True)


def test_compare_rejects_different_workloads():
    with pytest.raises(ValueError, match='workload'):
        compare({'workload_sha256':'a','metrics':{}}, {'workload_sha256':'b','metrics':{}})


def test_compare_computes_directions():
    baseline = {'workload_sha256':'a','metrics':{'output_tokens_per_second':10,'ttft_p95_ms':100}}
    candidate = {'workload_sha256':'a','metrics':{'output_tokens_per_second':12,'ttft_p95_ms':80}}
    result = compare(baseline, candidate)
    assert result['output_tokens_per_second'] == pytest.approx(20)
    assert result['ttft_p95_ms'] == pytest.approx(20)


def test_http_stream_end_to_end():
    import asyncio
    import httpx
    from benchmarks.serve import run_one
    async def exercise():
        async def handler(request):
            body = json.loads(request.content)
            assert body['stream_options']['include_usage'] is True
            stream = ('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                      'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                      'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
                      'data: [DONE]\n\n')
            return httpx.Response(200, text=stream)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_one(client, 'http://example/v1/chat/completions',
                                 {'id':'1','messages':[{'role':'user','content':'hi'}]}, 'm', 10, 2)
    result = asyncio.run(exercise())
    assert result['error'] is None
    assert result['completion_tokens'] == 2
    assert result['ttft_ms'] is not None


def test_compare_function_rejects_configuration_mismatch():
    base = {'workload_sha256': 'same', 'configuration': {'model': 'a'}, 'metrics': {}}
    other = dict(base, configuration={'model': 'b'})
    with pytest.raises(ValueError, match='configuration'):
        compare(base, other)
