import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


class Tokenizer:
    eos_token_id = 9
    vocab_size = 32

    def __init__(self):
        self.thread = threading.get_ident()

    def __len__(self):
        return 32

    def encode(self, text):
        assert threading.get_ident() == self.thread
        return [1] * len(text)

    def decode(self, ids, **kwargs):
        assert threading.get_ident() == self.thread
        return ''.join({7: 'A', 8: 'B', 9: ''}[i] for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert threading.get_ident() == self.thread
        assert tokenize and add_generation_prompt
        return [2] * sum(len(m['content']) for m in messages)


class Runner:
    def __init__(self, fail=False):
        self.thread = threading.get_ident()
        self.fail = fail
        self.batches = []
        self.exited = False

    def call(self, method, *args):
        assert threading.get_ident() == self.thread
        if method == 'exit':
            self.exited = True
            return
        seqs, _ = args
        if self.fail:
            raise RuntimeError('private GPU failure details')
        time.sleep(0.01)  # model execution only; scheduling remains real
        self.batches.append([s.seq_id for s in seqs])
        return [7 if s.num_completion_tokens == 0 else 9 for s in seqs]


def make_engine(fail=False):
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(max_model_len=32)
    engine.tokenizer = Tokenizer()
    engine.model_runner = Runner(fail)
    engine.ps = []
    engine.scheduler = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=4,
        eos=9, kvcache_block_size=256, num_kvcache_blocks=16))
    return engine


def test_engine_request_ids_validation_cancellation_and_idempotent_exit():
    engine = make_engine()
    seq_id = engine.add_request([1, 2, 3, 4, 5], SamplingParams(max_tokens=2))
    assert isinstance(seq_id, int)
    assert engine.abort_request(seq_id)
    assert not engine.abort_request(seq_id)
    assert engine.is_finished()
    seq_id = engine.add_request([1] * 5, SamplingParams(max_tokens=2))
    engine.step()  # partially prefills, owns KV pages
    assert engine.abort_request(seq_id)
    assert len(engine.scheduler.block_manager.free_block_ids) == 16
    for prompt in [[], [-1], [32]]:
        with pytest.raises(ValueError):
            engine.add_request(prompt, SamplingParams(max_tokens=1))
    with pytest.raises(ValueError):
        engine.add_request([1] * 32, SamplingParams(max_tokens=2))
    runner = engine.model_runner
    engine.exit()
    engine.exit()
    assert runner.exited


def test_api_completion_chat_stream_and_errors():
    from nanovllm.server.app import create_app

    async def run():
        app = create_app(engine_factory=make_engine, served_model_name='tiny')
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/v1/completions', json={'model':'tiny', 'prompt':'Hi', 'max_tokens':2})
                assert response.status_code == 200
                data = response.json()
                assert data['object'] == 'text_completion'
                assert data['choices'][0]['text'] == 'A'
                assert data['choices'][0]['finish_reason'] == 'stop'
                assert data['usage'] == {'prompt_tokens':2, 'completion_tokens':2, 'total_tokens':4}
                response = await client.post('/v1/chat/completions', json={'model':'tiny',
                    'messages':[{'role':'user','content':'Hi'}], 'max_tokens':1, 'temperature':0})
                assert response.status_code == 200
                data = response.json()
                assert data['choices'][0]['message'] == {'role':'assistant', 'content':'A'}
                assert data['choices'][0]['finish_reason'] == 'length'
                tiny = await client.post('/v1/completions', json={'model':'tiny','prompt':'a','max_tokens':1,'temperature':1e-12})
                assert tiny.status_code == 200
                response = await client.post('/v1/chat/completions', json={'model':'tiny',
                    'messages':[{'role':'user','content':'Hi'}], 'max_tokens':2, 'stream':True,
                    'stream_options':{'include_usage':True}})
                assert response.headers['content-type'].startswith('text/event-stream')
                frames = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
                assert frames[-1] == '[DONE]'
                chunks = [json.loads(frame) for frame in frames[:-1]]
                assert chunks[0]['choices'][0]['delta']['role'] == 'assistant'
                assert ''.join(c['choices'][0]['delta'].get('content','') for c in chunks if c['choices']) == 'A'
                assert chunks[-2]['choices'][0]['finish_reason'] == 'stop'
                assert chunks[-1]['choices'] == [] and chunks[-1]['usage']['completion_tokens'] == 2
                assert len({c['id'] for c in chunks}) == 1
                for payload in [
                    {'model':'other','prompt':'a'}, {'model':'tiny','prompt':''},
                    {'model':'tiny','prompt':'a','max_tokens':0},
                    {'model':'tiny','prompt':'a','temperature':-1},
                    {'model':'tiny','prompt':'a','n':2},
                    {'model':'tiny','prompt':'a','tools':[]},
                    {'model':'tiny','prompt':'a','priority':2},
                    {'model':'tiny','prompt':'a'*32,'max_tokens':1},
                ]:
                    error = await client.post('/v1/completions', json=payload)
                    assert error.status_code in (400,404)
                    assert 'message' in error.json()['error']
                assert (await client.get('/v1/models')).json()['data'][0]['id'] == 'tiny'
    asyncio.run(run())


def test_async_worker_batches_concurrent_requests_and_releases_cancelled_cache():
    from nanovllm.server.async_engine import AsyncLLMEngine

    async def run():
        backend = AsyncLLMEngine(make_engine)
        await backend.start()
        try:
            a, b = await asyncio.gather(
                backend.add_request('abcdefgh', SamplingParams(max_tokens=3)),
                backend.add_request('abc', SamplingParams(max_tokens=3)))
            assert a.seq_id != b.seq_id
            await backend.cancel(a)
            events = []
            while not events or not events[-1]['finished']:
                events.append(await asyncio.wait_for(b.queue.get(), timeout=5))
            assert [i for e in events for i in e['token_ids']] == [7, 9]
            await backend.cancel(b)
            free = await backend.call(lambda: len(backend.engine.scheduler.block_manager.free_block_ids))
            assert free == 16
            c, d = await asyncio.gather(
                backend.add_request('a', SamplingParams(max_tokens=2)),
                backend.add_request('b', SamplingParams(max_tokens=2)))
            for handle in (c, d):
                while not (await handle.queue.get())['finished']:
                    pass
            batches = await backend.call(lambda: backend.engine.model_runner.batches)
            assert any(c.seq_id in batch and d.seq_id in batch for batch in batches)
        finally:
            await backend.close()
            await backend.close()
    asyncio.run(run())


def test_model_failure_returns_openai_error_and_shutdown_is_clean():
    from nanovllm.server.app import create_app

    async def run():
        app = create_app(engine_factory=lambda: make_engine(fail=True), served_model_name='tiny')
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                result = await client.post('/v1/completions', json={'model':'tiny','prompt':'a','max_tokens':1})
                assert result.status_code == 500
                assert result.json()['error']['type'] == 'server_error'
                assert 'private GPU' not in result.text
                assert (await client.get('/health')).status_code == 503
                retry = await client.post('/v1/completions', json={'model':'tiny','prompt':'a','max_tokens':1})
                assert retry.status_code == 503
    asyncio.run(run())


def test_backpressure_shutdown_and_cancelled_admission_release_all_requests():
    from nanovllm.server.async_engine import AsyncLLMEngine
    from nanovllm.server.errors import APIError

    async def run():
        backend = AsyncLLMEngine(make_engine, event_queue_size=1)
        await backend.start()
        handle = await backend.add_request('a', SamplingParams(max_tokens=3))
        # Wait behind worker steps without consuming the request's single slot.
        for _ in range(100):
            if handle.seq_id not in backend.pending:
                break
            await asyncio.sleep(0.01)
        assert isinstance(await handle.queue.get(), APIError)
        assert await backend.call(lambda: backend.engine.is_finished())
        # Admission cancellation can race tokenization; the new sequence must
        # be cancelled even when admission returns after the caller is gone.
        task = asyncio.create_task(backend.add_request('a' * 20, SamplingParams(max_tokens=2)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        live = await backend.add_request('a' * 20, SamplingParams(max_tokens=2))
        await backend.close()
        assert isinstance(await live.queue.get(), APIError)
        assert not backend.pending
    asyncio.run(run())


def test_completion_stream_holds_incomplete_utf8_and_reports_model_errors():
    from nanovllm.server.app import create_app

    class UTF8Tokenizer(Tokenizer):
        def decode(self, ids, **kw):
            return '\ufffd' if ids == [7] else '你'

    async def run():
        def factory():
            engine = make_engine()
            engine.tokenizer = UTF8Tokenizer()
            return engine
        for fail in (False, True):
            app = create_app(engine_factory=(lambda: make_engine(True)) if fail else factory, served_model_name='tiny')
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    result = await client.post('/v1/completions', json={'model':'tiny','prompt':[1], 'max_tokens':2, 'stream':True})
                    frames = [json.loads(line[6:]) for line in result.text.splitlines() if line.startswith('data: {')]
                    if fail:
                        assert frames[-1]['error']['type'] == 'server_error'
                        assert 'private GPU' not in result.text
                    else:
                        assert ''.join(frame['choices'][0]['text'] for frame in frames) == '你'
                    assert result.text.endswith('data: [DONE]\n\n')
    asyncio.run(run())


def test_real_http_sse_arrives_before_generation_finishes_and_disconnect_cancels():
    import socket
    import uvicorn
    from nanovllm.server.app import create_app

    release_decode = threading.Event()
    seen = []

    class BlockingRunner(Runner):
        def call(self, method, *args):
            if method == 'run' and any(s.num_completion_tokens for s in args[0]):
                if not release_decode.wait(timeout=10):
                    raise RuntimeError('test did not release decode')
            result = super().call(method, *args)
            return [7] * len(args[0]) if method == 'run' else result

    def factory():
        engine = make_engine()
        engine.model_runner = BlockingRunner()
        seen.append((engine, engine.model_runner))
        return engine

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    app = create_app(engine_factory=factory, served_model_name='tiny')
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    thread = threading.Thread(target=server.run, kwargs={'sockets':[sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5) as client:
            with client.stream('POST', '/v1/completions', json={'model':'tiny','prompt':'a','max_tokens':16,'stream':True}) as response:
                first = next(line for line in response.iter_lines() if line.startswith('data: '))
                assert json.loads(first[6:])['choices'][0]['text'] == 'A'
                assert not release_decode.is_set()  # first token arrived before decode completes
                assert client.get('/health').status_code == 200  # event loop stays responsive
            release_decode.set()  # disconnected stream must cancel, not wait for full generation
            deadline = time.monotonic() + 5
            while app.state.backend.pending and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not app.state.backend.pending
            assert len(seen[0][1].batches) < 16
    finally:
        release_decode.set()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
    assert not thread.is_alive()
    assert not hasattr(seen[0][0], 'model_runner')  # lifespan closed the engine


def test_server_help_does_not_require_cuda_dependencies():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-m', 'nanovllm.server', '--help'], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and '--quantization' in result.stdout
    for option in ['--scheduling-policy', '--long-prefill-token-threshold',
                   '--long-prefill-token-threshold-adaptive', '--no-scheduler-reserve-full-isl']:
        assert option in result.stdout


def test_disconnect_before_stream_body_starts_reclaims_prefill():
    from starlette.requests import ClientDisconnect
    from nanovllm.server.app import create_app

    async def run():
        app = create_app(engine_factory=make_engine, served_model_name='tiny')
        async with app.router.lifespan_context(app):
            body = json.dumps({'model':'tiny','prompt':'a'*20,'max_tokens':2,'stream':True}).encode()
            scope = {'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},
                'http_version':'1.1','method':'POST','scheme':'http','path':'/v1/completions',
                'raw_path':b'/v1/completions','query_string':b'',
                'headers':[(b'content-type',b'application/json')], 'client':('127.0.0.1',1),'server':('test',80)}
            async def receive():
                return {'type':'http.request','body':body,'more_body':False}
            async def send(message):
                if message['type'] == 'http.response.start':
                    raise OSError('peer disconnected before stream started')
            with pytest.raises(ClientDisconnect):
                await app(scope, receive, send)
            assert not app.state.backend.pending
            assert await app.state.backend.call(app.state.backend.engine.is_finished)
    asyncio.run(run())


def test_openai_sdk_accepts_completion_and_chat_stream():
    openai = pytest.importorskip('openai')
    from nanovllm.server.app import create_app

    async def run():
        app = create_app(engine_factory=make_engine, served_model_name='tiny')
        async with app.router.lifespan_context(app):
            async with openai.AsyncOpenAI(api_key='unused', base_url='http://test/v1',
                http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app))) as client:
                response = await client.completions.create(model='tiny', prompt='a', max_tokens=2)
                assert response.choices[0].text == 'A' and response.usage.total_tokens == 3
                response = await client.chat.completions.create(model='tiny',
                    messages=[{'role':'user','content':'a'}], max_tokens=2, stream=True,
                    stream_options={'include_usage':True})
                chunks = [chunk async for chunk in response]
                assert ''.join(c.choices[0].delta.content or '' for c in chunks if c.choices) == 'A'
                assert chunks[-1].usage.total_tokens == 3
    asyncio.run(run())


def test_api_priority_and_async_admission_propagate_to_engine():
    from nanovllm.server.app import create_app
    seen = []

    class PriorityEngine(LLMEngine):
        def add_request(self, prompt, params, *, priority=0):
            seen.append(priority)
            return super().add_request(prompt, params, priority=priority)

    def factory():
        engine = make_engine()
        engine.__class__ = PriorityEngine
        engine.scheduler.policy = 'priority'
        return engine

    async def run():
        app = create_app(engine_factory=factory, served_model_name='tiny')
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                result = await client.post('/v1/completions', json={'model':'tiny', 'prompt':'a', 'max_tokens':1, 'priority':-3})
                assert result.status_code == 200 and seen == [-3]
                result = await client.post('/v1/chat/completions', json={'model':'tiny',
                    'messages':[{'role':'user','content':'a'}], 'max_tokens':1, 'priority':2})
                assert result.status_code == 200 and seen == [-3, 2]
                result = await client.post('/v1/completions', json={'model':'tiny','prompt':'a','max_tokens':1,'priority':True})
                assert result.status_code == 400 and seen == [-3, 2]
    asyncio.run(run())
