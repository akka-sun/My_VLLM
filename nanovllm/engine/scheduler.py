from collections import deque

from nanovllm.config import Config, validate_scheduler_config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.speculative import SpeculativeResult


class Scheduler:

    def __init__(self, config: Config):
        validate_scheduler_config(config)
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.policy = getattr(config, "scheduling_policy", "fcfs")
        self.long_prefill_token_threshold = getattr(config, "long_prefill_token_threshold", 0)
        self.adaptive_threshold = getattr(config, "long_prefill_token_threshold_adaptive", False)
        self.reserve_full_isl = getattr(config, "scheduler_reserve_full_isl", True)
        self.speculative_config = getattr(config, "speculative_config", None)
        self.num_preemptions = 0
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.num_tokens > len(self.block_manager.blocks) * self.block_size:
            raise ValueError("sequence exceeds total KV cache capacity")
        self._enqueue(seq)

    def _enqueue(self, seq: Sequence, *, preempted=False):
        if self.policy == "priority":
            key = (seq.priority, seq.seq_id)
            index = next((i for i, other in enumerate(self.waiting)
                          if (other.priority, other.seq_id) > key), len(self.waiting))
            self.waiting.insert(index, seq)
        elif preempted:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    self.block_manager.deallocate(seq)
                    seq.num_scheduled_tokens = 0
                    seq.status = SequenceStatus.FINISHED
                    return True
        return False

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Advance RUNNING token cursors, then admit WAITING requests.

        RUNNING includes partial prefills and decode. The returned bool only
        selects the existing varlen-vs-decode execution path.
        """
        scheduled_seqs = []
        budget = self.max_num_batched_tokens
        eligible = len(self.running) + len(self.waiting)
        threshold = self.long_prefill_token_threshold if eligible > 1 else 0
        if threshold and self.adaptive_threshold:
            threshold = max(threshold, budget // eligible)

        def chunk_size(pending):
            return min(pending, budget, threshold or pending)

        def record(seq, count):
            # A one-token recompute chunk may still need historical token IDs
            # in TP workers, so derive this from progress, not chunk size.
            seq.is_prefill = seq.num_computed_tokens < seq.num_prompt_tokens or seq.num_pending_tokens > 1
            seq.num_scheduled_tokens = count
            scheduled_seqs.append(seq)

        def allocate_running(seq):
            count = chunk_size(seq.num_pending_tokens)
            seq.num_speculative_tokens = 0
            if (self.speculative_config and seq.num_pending_tokens == 1
                    and seq.num_computed_tokens >= seq.num_prompt_tokens):
                seq.num_speculative_tokens = min(
                    self.speculative_config["num_speculative_tokens"], budget - 1,
                    seq.max_tokens - seq.num_completion_tokens - 1)
                count = 1 + seq.num_speculative_tokens
            allocated = self.block_manager.allocate_slots(
                seq, count, lookahead=seq.num_speculative_tokens)
            # Under page pressure prefer a shorter block to preemption.
            while allocated is None and seq.num_speculative_tokens:
                seq.num_speculative_tokens -= 1
                count -= 1
                allocated = self.block_manager.allocate_slots(
                    seq, count, lookahead=seq.num_speculative_tokens)
            return count, allocated

        preempted = False
        index = 0
        while index < len(self.running) and budget > 0:
            seq = self.running[index]
            count, allocated = allocate_running(seq)
            while allocated is None:
                victim = (max(self.running, key=lambda s: (s.priority, s.seq_id))
                          if self.policy == "priority" else self.running[-1])
                victim_index = self.running.index(victim)
                if victim in scheduled_seqs:
                    scheduled_seqs.remove(victim)
                    budget += victim.num_scheduled_tokens
                self.running.remove(victim)
                if victim_index < index:
                    index -= 1
                self.preempt(victim)
                preempted = True
                if victim is seq:
                    break
                count, allocated = allocate_running(seq)
            if allocated is None:
                if self.policy == "priority" and seq.status == SequenceStatus.PREEMPTED:
                    continue
                break
            record(seq, count)
            budget -= count
            index += 1

        # Re-admission on a preempting step would immediately reclaim pages
        # just released to let RUNNING work progress.
        while not preempted and self.waiting and len(self.running) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            # DSpark needs the auxiliary features for every prefix position;
            # target KV-only prefix hits cannot reconstruct those features.
            needs_aux = self.speculative_config and self.speculative_config['method'] == 'dspark'
            hits = [] if needs_aux else self.block_manager.get_computed_blocks(seq)
            count = chunk_size(seq.num_tokens - len(hits) * self.block_size)
            allocated = self.block_manager.allocate_slots(seq, count, hits,
                full_sequence_must_fit=self.reserve_full_isl)
            if allocated is None:
                break
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            record(seq, count)
            budget -= count
        return scheduled_seqs, any(seq.is_prefill for seq in scheduled_seqs)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.PREEMPTED
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        seq.num_speculative_tokens = 0
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.block_manager.deallocate(seq)
        self._enqueue(seq, preempted=True)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if isinstance(token_id, SpeculativeResult):
                old_length = seq.num_tokens
                for token in token_id.token_ids:
                    seq.append_token(token)
                    if ((not seq.ignore_eos and token == self.eos)
                            or seq.num_completion_tokens >= seq.max_tokens):
                        break
                # One pending correction/bonus; never publish an unverified hash.
                seq.num_scheduled_tokens = seq.num_tokens - old_length
                self.block_manager.hash_blocks(seq)
                seq.num_computed_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = seq.num_speculative_tokens = 0
                self.block_manager.truncate(seq)
                if ((not seq.ignore_eos and seq.last_token == self.eos)
                        or seq.num_completion_tokens >= seq.max_tokens):
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_computed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_computed_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
