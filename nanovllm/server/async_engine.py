"""One engine-owning thread; async command admission and bounded event queues."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
import logging

from nanovllm.server.errors import APIError

logger = logging.getLogger(__name__)


@dataclass
class RequestHandle:
    seq_id: int
    prompt_tokens: int
    queue: asyncio.Queue = field(repr=False)
    token_ids: list[int] = field(default_factory=list, repr=False)


class AsyncLLMEngine:
    def __init__(self, engine_factory, max_pending_requests=256, event_queue_size=128):
        if max_pending_requests < 1 or event_queue_size < 1:
            raise ValueError('queue limits must be positive')
        self.factory = engine_factory
        self.max_pending_requests = max_pending_requests
        self.event_queue_size = event_queue_size
        self.commands = asyncio.Queue(maxsize=max_pending_requests * 2)
        self.pending = {}
        self.engine = None
        self.executor = None
        self.task = None
        self.closing = False
        self.failed = False

    async def call(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self.executor, partial(fn, *args, **kwargs))

    async def start(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nanovllm-engine')
        try:
            self.engine = await self.call(self.factory)
        except BaseException:
            self.executor.shutdown(wait=True)
            self.executor = None
            raise
        self.task = asyncio.create_task(self._run(), name='nanovllm-scheduler')

    async def add_request(self, prompt, sampling_params, *, chat=False, priority=0):
        if self.closing or self.failed or self.task is None or self.task.done():
            raise APIError('Inference service is unavailable', 503, 'server_error')
        future = asyncio.get_running_loop().create_future()
        try:
            self.commands.put_nowait(('add', (prompt, sampling_params, chat, priority), future))
        except asyncio.QueueFull:
            raise APIError('Request queue is full', 503, 'server_error') from None
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Admission may already be running in the worker thread. Cancel the
            # resulting sequence as soon as its ID is known.
            def cleanup(done):
                if not done.cancelled() and done.exception() is None:
                    asyncio.create_task(self.cancel(done.result()))
            future.add_done_callback(cleanup)
            raise

    async def cancel(self, handle):
        if self.closing or self.task is None or self.task.done():
            return
        future = asyncio.get_running_loop().create_future()
        await self.commands.put(('cancel', handle, future))
        await asyncio.shield(future)

    def _add(self, prompt, params, chat, priority):
        if chat:
            try:
                token_ids = self.engine.tokenizer.apply_chat_template(
                    prompt, tokenize=True, add_generation_prompt=True)
            except (ValueError, TypeError) as exc:
                raise ValueError('Tokenizer cannot apply the requested chat template') from exc
        else:
            token_ids = self.engine.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        seq_id = self.engine.add_request(token_ids, params, priority=priority)
        return seq_id, len(token_ids)

    @staticmethod
    def _terminal(handle, error):
        while not handle.queue.empty():
            handle.queue.get_nowait()
        handle.queue.put_nowait(error)

    async def _command(self, command):
        kind, payload, future = command
        if kind == 'close':
            return False
        if kind == 'cancel':
            handle = self.pending.pop(payload.seq_id, None)
            if handle is not None:
                await self.call(self.engine.abort_request, handle.seq_id)
                self._terminal(handle, APIError('Request cancelled', 503, 'server_error'))
            if not future.done():
                future.set_result(None)
            return True
        if self.failed or self.closing:
            future.set_exception(APIError('Inference service is unavailable', 503, 'server_error'))
        elif len(self.pending) >= self.max_pending_requests:
            future.set_exception(APIError('Too many active requests', 503, 'server_error'))
        else:
            try:
                seq_id, count = await self.call(self._add, *payload)
                handle = RequestHandle(seq_id, count, asyncio.Queue(maxsize=self.event_queue_size))
                self.pending[seq_id] = handle
                future.set_result(handle)
            except ValueError as exc:
                future.set_exception(APIError(str(exc)))
            except Exception:
                logger.exception('Request admission failed')
                future.set_exception(APIError('Request admission failed', 500, 'server_error'))
        return True

    async def _run(self):
        try:
            while True:
                if not self.pending:
                    if not await self._command(await self.commands.get()):
                        break
                for _ in range(64):
                    try:
                        command = self.commands.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if not await self._command(command):
                        return
                if not self.pending:
                    continue
                try:
                    outputs, _ = await self.call(self.engine.step, stream=True)
                except Exception:
                    logger.exception('Model execution failed')
                    self.failed = True
                    await self._abort_all(APIError('Model execution failed', 500, 'server_error'))
                    continue
                for output in outputs:
                    handle = self.pending.get(output['seq_id'])
                    if handle is None:
                        continue
                    handle.token_ids.extend(output['token_ids'])
                    try:
                        # Decode before launching the next GPU step. Otherwise
                        # that step can block token delivery on this executor.
                        text = await self.decode(handle.token_ids)
                    except Exception:
                        logger.exception('Token decoding failed')
                        await self.call(self.engine.abort_request, handle.seq_id)
                        self._terminal(handle, APIError('Token decoding failed', 500, 'server_error'))
                        self.pending.pop(handle.seq_id, None)
                        continue
                    output = {**output, 'text':text}
                    try:
                        handle.queue.put_nowait(output)
                    except asyncio.QueueFull:
                        await self.call(self.engine.abort_request, handle.seq_id)
                        self._terminal(handle, APIError('Client is consuming too slowly', 503, 'server_error'))
                        self.pending.pop(handle.seq_id, None)
                    if output['finished']:
                        self.pending.pop(handle.seq_id, None)
        finally:
            await self._abort_all(APIError('Inference service is shutting down', 503, 'server_error'))
            while not self.commands.empty():
                _, _, future = self.commands.get_nowait()
                if future is not None and not future.done():
                    future.set_exception(APIError('Inference service is shutting down', 503, 'server_error'))

    async def _abort_all(self, error):
        for handle in list(self.pending.values()):
            try:
                await self.call(self.engine.abort_request, handle.seq_id)
            except Exception:
                logger.exception('Request cancellation failed')
            self._terminal(handle, error)
        self.pending.clear()

    async def decode(self, token_ids):
        return await self.call(self.engine.tokenizer.decode, token_ids,
            skip_special_tokens=True, clean_up_tokenization_spaces=False)

    async def close(self):
        if self.executor is None:
            return
        self.closing = True
        try:
            if self.task is not None and not self.task.done():
                await self.commands.put(('close', None, None))
                await self.task
            if self.engine is not None:
                await self.call(self.engine.exit)
        finally:
            self.executor.shutdown(wait=True)
            self.executor = None
