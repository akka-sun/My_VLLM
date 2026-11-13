import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return
        del self.model_runner
        atexit.unregister(self.exit)
        try:
            runner.call("exit")
        finally:
            for p in self.ps:
                p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, *, priority: int = 0):
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("priority must be an integer")
        if priority != 0 and self.scheduler.policy != "priority":
            raise ValueError("nonzero priority requires scheduling_policy='priority'")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in prompt):
            raise ValueError("prompt must contain non-negative integer token IDs")
        if hasattr(self.tokenizer, "__len__") and any(t >= len(self.tokenizer) for t in prompt):
            raise ValueError("prompt contains an out-of-vocabulary token ID")
        if isinstance(sampling_params.max_tokens, bool) or not isinstance(sampling_params.max_tokens, int) or sampling_params.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        config = getattr(self, "config", None)
        if config is not None and len(prompt) + sampling_params.max_tokens > config.max_model_len:
            raise ValueError("prompt plus max_tokens exceeds max_model_len")
        capacity = len(self.scheduler.block_manager.blocks) * self.scheduler.block_size
        if len(prompt) + sampling_params.max_tokens - 1 > capacity:
            raise ValueError("request exceeds total KV cache capacity")
        seq = Sequence(prompt, sampling_params, priority=priority)
        self.scheduler.add(seq)
        return seq.seq_id

    def abort_request(self, seq_id: int) -> bool:
        """Remove a queued/active request and release its KV cache pages."""
        aborted = self.scheduler.abort(seq_id)
        if aborted and getattr(getattr(self, 'config', None), 'speculative_config', None):
            self.model_runner.call('retain_requests', [s.seq_id for s in self.scheduler.running])
        return aborted

    def step(self, stream: bool = False):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        previous_lengths = [seq.num_completion_tokens for seq in seqs]
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        speculative = getattr(getattr(self, 'config', None), 'speculative_config', None)
        if speculative:
            self.model_runner.call('retain_requests', [s.seq_id for s in self.scheduler.running])
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if speculative:
            self.model_runner.call('retain_requests', [s.seq_id for s in self.scheduler.running])
        if not is_prefill:
            num_tokens = -sum(seq.num_completion_tokens - previous for seq, previous in zip(seqs, previous_lengths))
        if stream:
            outputs = [
                {"seq_id": seq.seq_id, "token_ids": seq.completion_token_ids[previous_length:],
                 "finished": seq.is_finished}
                for seq, previous_length in zip(seqs, previous_lengths)
                if seq.num_completion_tokens > previous_length
            ]
        else:
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate_stream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        """Yield token deltas and cumulative text after each step.

        Exhaust the iterator before starting another generation on this engine.
        Closing it early leaves pending requests queued for subsequent steps.
        Cumulative text may revise its suffix as the tokenizer resolves bytes.
        """
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params must match the number of prompts")
        if not self.is_finished():
            raise ValueError("generate_stream requires an engine with no pending requests")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        completions = {}
        while not self.is_finished():
            outputs, _ = self.step(stream=True)
            for output in outputs:
                token_ids = completions.setdefault(output["seq_id"], [])
                token_ids.extend(output["token_ids"])
                yield {**output, "text": self.tokenizer.decode(token_ids)}

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
