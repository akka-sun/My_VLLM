"""Text-only OpenAI-compatible completions and chat completions."""
import asyncio
import anyio
from contextlib import asynccontextmanager
import json
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException

from nanovllm.sampling_params import SamplingParams
from nanovllm.server.async_engine import AsyncLLMEngine
from nanovllm.server.errors import APIError
from nanovllm.server.schemas import CompletionRequest, ChatCompletionRequest


class CancellableStreamingResponse(StreamingResponse):
    def __init__(self, *args, cleanup, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A generator's finally is never entered if response.start fails.
            # Shield against Starlette's disconnect cancellation scope as well.
            with anyio.CancelScope(shield=True):
                await self.cleanup()


def create_app(model=None, *, served_model_name='nano-vllm', engine_factory=None,
               engine_kwargs=None, max_pending_requests=256):
    if engine_factory is None:
        if model is None:
            raise ValueError('model or engine_factory is required')

        def engine_factory():
            from nanovllm.engine.llm_engine import LLMEngine
            return LLMEngine(model, **(engine_kwargs or {}))

    @asynccontextmanager
    async def lifespan(app):
        backend = AsyncLLMEngine(engine_factory, max_pending_requests)
        await backend.start()
        app.state.backend = backend
        try:
            yield
        finally:
            await backend.close()

    app = FastAPI(title='Nano-vLLM OpenAI-compatible API', lifespan=lifespan)

    @app.exception_handler(APIError)
    async def api_error(request, error):
        return JSONResponse(error.body, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        fields = ['.'.join(str(p) for p in item['loc']) + ': ' + item['msg'] for item in error.errors()]
        return JSONResponse(APIError('; '.join(fields)).body, status_code=400)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse(APIError(str(error.detail), error.status_code).body, status_code=error.status_code)

    @app.get('/health')
    async def health():
        backend = app.state.backend
        if backend.failed or backend.closing or backend.task.done():
            raise APIError('Inference service is unavailable', 503, 'server_error')
        return {'status':'ok'}

    @app.get('/v1/models')
    async def models():
        return {'object':'list', 'data':[{'id':served_model_name, 'object':'model', 'created':0, 'owned_by':'nanovllm'}]}

    async def respond(body, request, chat):
        if body.model != served_model_name:
            raise APIError(f'Model {body.model!r} is not served', 404, param='model')
        if body.stream_options is not None and not body.stream:
            raise APIError('stream_options requires stream=true', param='stream_options')
        greedy = body.temperature <= 1e-10
        params = SamplingParams(temperature=1.0 if greedy else body.temperature,
            top_k=1 if greedy else body.top_k,
            top_p=body.top_p, max_tokens=body.max_tokens)
        backend = app.state.backend
        prompt = [message.model_dump() for message in body.messages] if chat else body.prompt
        handle = await backend.add_request(prompt, params, chat=chat)
        response_id = ('chatcmpl-' if chat else 'cmpl-') + uuid4().hex
        created = int(time.time())
        base = {'id':response_id, 'object':'chat.completion.chunk' if chat else 'text_completion',
                'created':created, 'model':served_model_name}

        def usage(count):
            return {'prompt_tokens':handle.prompt_tokens, 'completion_tokens':count,
                    'total_tokens':handle.prompt_tokens + count}

        async def events():
            # Poll disconnects even during a long prefill with no emitted token.
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(handle.queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if isinstance(event, APIError):
                    raise event
                yield event
                if event['finished']:
                    return

        def reason(ids):
            eos = backend.engine.tokenizer.eos_token_id
            return 'stop' if ids and ids[-1] == eos else 'length'

        def frame(value):
            return 'data: ' + json.dumps(value, ensure_ascii=False) + '\n\n'

        async def stream():
            ids = []
            sent = ''
            finished = False
            include_usage = body.stream_options is not None and body.stream_options.include_usage
            try:
                if chat:
                    first = {**base, 'choices':[{'index':0, 'delta':{'role':'assistant','content':''}, 'finish_reason':None}]}
                    if include_usage:
                        first['usage'] = None
                    yield frame(first)
                async for event in events():
                    ids.extend(event['token_ids'])
                    text = event['text']
                    # Incomplete byte-fallback tokens often decode to a trailing
                    # replacement character; wait for their remaining bytes.
                    if not event['finished']:
                        text = text.rstrip('\ufffd')
                    if not text.startswith(sent):
                        raise APIError('Tokenizer revised previously streamed text', 500, 'server_error')
                    delta, sent = text[len(sent):], text
                    choice = {'index':0, 'finish_reason':reason(ids) if event['finished'] else None}
                    choice['delta' if chat else 'text'] = {'content':delta} if chat else delta
                    chunk = {**base, 'choices':[choice]}
                    if include_usage:
                        chunk['usage'] = None
                    yield frame(chunk)
                    finished = event['finished']
                if not finished:
                    return
                if include_usage:
                    yield frame({**base, 'choices':[], 'usage':usage(len(ids))})
                yield 'data: [DONE]\n\n'
            except APIError as error:
                yield frame(error.body)
                yield 'data: [DONE]\n\n'

        if body.stream:
            return CancellableStreamingResponse(stream(), cleanup=lambda: backend.cancel(handle), media_type='text/event-stream',
                headers={'Cache-Control':'no-cache', 'X-Accel-Buffering':'no'})
        ids = []
        finished = False
        text = ''
        try:
            async for event in events():
                ids.extend(event['token_ids'])
                finished, text = event['finished'], event['text']
            if not finished:
                raise APIError('Client disconnected', 499, 'server_error')
            choice = {'index':0, 'finish_reason':reason(ids)}
            if chat:
                choice['message'] = {'role':'assistant', 'content':text}
            else:
                choice.update(text=text, logprobs=None)
            return {**base, 'object':'chat.completion' if chat else 'text_completion',
                    'choices':[choice], 'usage':usage(len(ids))}
        finally:
            await asyncio.shield(backend.cancel(handle))

    @app.post('/v1/completions')
    async def completions(body: CompletionRequest, request: Request):
        return await respond(body, request, False)

    @app.post('/v1/chat/completions')
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        return await respond(body, request, True)

    return app
