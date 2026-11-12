from collections import Counter
from types import SimpleNamespace
import random
import pickle

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def scheduler(budget=256, blocks=8, slots=8, **options):
    config = dict(max_num_seqs=slots, max_num_batched_tokens=budget,
        eos=99, kvcache_block_size=256, num_kvcache_blocks=blocks)
    return Scheduler(SimpleNamespace(**(config | options)))


def tick(sched):
    seqs, prefill = sched.schedule()
    counts = [s.num_scheduled_tokens for s in seqs]
    sched.postprocess(seqs, [7] * len(seqs), prefill)
    return seqs, counts, prefill


def test_partial_prefill_is_running_with_unified_progress():
    sched = scheduler(budget=2)
    seq = Sequence([1] * 6, SamplingParams(max_tokens=1))
    sched.add(seq)
    tick(sched)
    assert list(sched.running) == [seq] and not sched.waiting
    assert seq.status == SequenceStatus.RUNNING
    assert seq.num_computed_tokens == 2 and seq.num_pending_tokens == 4
    assert seq.num_cached_tokens == 2 and seq.num_completion_tokens == 0


@pytest.mark.parametrize('adaptive, first_count', [(False, 1), (True, 4)])
def test_long_threshold_shares_budget_and_lone_request_uses_whole_budget(adaptive, first_count):
    sched = scheduler(budget=8, long_prefill_token_threshold=1,
        long_prefill_token_threshold_adaptive=adaptive)
    long = Sequence([1] * 20, SamplingParams(max_tokens=1))
    short = Sequence([2], SamplingParams(max_tokens=1))
    sched.add(long)
    sched.add(short)
    seqs, counts, _ = tick(sched)
    assert seqs == [long, short] and counts == [first_count, 1]
    assert short.is_finished
    assert tick(sched)[1] == [8]


def test_running_partial_prefill_and_decode_follow_admission_order():
    sched = scheduler(budget=4, long_prefill_token_threshold=2)
    long = Sequence([1] * 8, SamplingParams(max_tokens=1))
    short = Sequence([2], SamplingParams(max_tokens=3))
    sched.add(long)
    sched.add(short)
    tick(sched)
    seqs, counts, prefill = tick(sched)
    assert seqs == [long, short] and counts == [2, 1] and prefill
    assert short.num_completion_tokens == 2


def test_prompt_kv_pages_grow_only_with_scheduled_progress():
    sched = scheduler(budget=128)
    seq = Sequence([1] * 1024, SamplingParams(max_tokens=1))
    sched.add(seq)
    tick(sched)
    assert len(seq.block_table) == 1 and len(sched.block_manager.free_block_ids) == 7
    tick(sched)
    assert len(seq.block_table) == 1
    tick(sched)
    assert len(seq.block_table) == 2 and seq.num_computed_tokens == 384


@pytest.mark.parametrize('reserve, expected_count', [(True, 1), (False, 2)])
def test_full_input_admission_gate_does_not_allocate_future_pages(reserve, expected_count):
    sched = scheduler(budget=256, blocks=3, scheduler_reserve_full_isl=reserve)
    a = Sequence([1] * 300, SamplingParams(max_tokens=2))
    b = Sequence([2] * 512, SamplingParams(max_tokens=1))
    sched.add(a)
    tick(sched)
    sched.add(b)
    seqs, _, _ = tick(sched)
    assert len(seqs) == expected_count
    assert len(a.block_table) == 2
    assert len(b.block_table) == (0 if reserve else 1)


def prime_prefix(manager):
    seq = Sequence([5] * 256 + [6], SamplingParams(max_tokens=1))
    assert manager.allocate_slots(seq, 256) is not None
    seq.num_scheduled_tokens = 256
    manager.hash_blocks(seq)
    seq.num_computed_tokens = 256
    return seq


def test_failed_prefix_adoption_is_atomic_and_free_hits_are_pinned_before_eviction():
    manager = BlockManager(2, 256)
    prime = prime_prefix(manager)
    other = Sequence([8], SamplingParams(max_tokens=1))
    assert manager.allocate_slots(other, 1) is not None
    cached_page = prime.block_table[0]
    manager.deallocate(prime)
    seq = Sequence([5] * 256 + [6] * 257, SamplingParams(max_tokens=1))
    hits = manager.get_computed_blocks(seq)
    assert hits == [cached_page]
    before = list(manager.free_block_ids)
    assert manager.allocate_slots(seq, 257, hits) is None
    assert seq.block_table == [] and seq.num_computed_tokens == 0
    assert list(manager.free_block_ids) == before and manager.blocks[cached_page].ref_count == 0
    manager.deallocate(other)  # free order is cached page first, then other page
    assert list(manager.free_block_ids)[0] == cached_page
    assert manager.allocate_slots(seq, 1, hits) is not None
    assert seq.block_table[0] == cached_page and len(seq.block_table) == 2
    assert manager.blocks[cached_page].token_ids == [5] * 256
    assert seq.num_computed_tokens == 256
    manager.deallocate(seq)
    assert len(manager.free_block_ids) == 2


def test_exact_full_prefix_hit_leaves_last_page_for_a_real_query():
    manager = BlockManager(4, 256)
    prime = Sequence([1] * 512, SamplingParams(max_tokens=1))
    manager.allocate_slots(prime, 512)
    prime.num_scheduled_tokens = 512
    manager.hash_blocks(prime)
    manager.deallocate(prime)
    request = Sequence([1] * 512, SamplingParams(max_tokens=1))
    assert len(manager.get_computed_blocks(request)) == 1


def test_priority_preemption_rolls_back_already_scheduled_victim_and_keeps_outputs():
    sched = scheduler(budget=512, blocks=3, scheduling_policy='priority')
    a = Sequence([1] * 256, SamplingParams(max_tokens=4), priority=10)
    b = Sequence([2] * 256, SamplingParams(max_tokens=4), priority=-1)
    sched.add(a)
    tick(sched)
    sched.add(b)
    tick(sched)
    assert list(sched.running) == [a, b]
    assert a.completion_token_ids == [7, 7]
    new = Sequence([3], SamplingParams(max_tokens=1), priority=-2)
    sched.add(new)
    seqs, counts, _ = tick(sched)
    assert seqs == [b] and counts == [1]
    assert a.num_scheduled_tokens == 0 and a.num_computed_tokens == 0
    assert a.num_preemptions == sched.num_preemptions == 1
    assert a.completion_token_ids == [7, 7]
    assert new in sched.waiting and a in sched.waiting  # no same-step re-admission
    for _ in range(30):
        if sched.is_finished():
            break
        tick(sched)
    assert sched.is_finished()
    assert a.completion_token_ids == b.completion_token_ids == [7] * 4


def test_rolled_back_token_budget_is_reused_by_later_running_work():
    sched = scheduler(budget=2, blocks=4, scheduling_policy='priority')
    a = Sequence([1] * 256, SamplingParams(max_tokens=5), priority=10)
    a.append_token(7)
    a.append_token(7)
    b = Sequence([2] * 256, SamplingParams(max_tokens=5), priority=0)
    b.append_token(7)
    c = Sequence([3], SamplingParams(max_tokens=5), priority=-1)
    c.append_token(7)
    # Valid states after completed steps: A has two pages, B and C one each.
    for seq, computed in [(a, 257), (b, 256), (c, 1)]:
        sched.block_manager.allocate_slots(seq, computed)
        seq.num_computed_tokens = computed
        seq.status = SequenceStatus.RUNNING
        sched.running.append(seq)
    seqs, counts, _ = tick(sched)
    # A was scheduled first, then evicted to extend B. Its restored budget
    # lets C run too; forgetting rollback would return B alone.
    assert seqs == [b, c] and counts == [1, 1]
    assert a.num_preemptions == 1 and a.num_scheduled_tokens == 0


def test_fcfs_preempts_latest_running_request_and_reuses_completed_prefix():
    sched = scheduler(budget=512, blocks=2)
    a = Sequence([1] * 256, SamplingParams(max_tokens=2))
    b = Sequence([2] * 256, SamplingParams(max_tokens=2))
    sched.add(a)
    sched.add(b)
    tick(sched)
    seqs, prefill = sched.schedule()
    assert seqs == [a] and not prefill
    assert b.status == SequenceStatus.PREEMPTED and b.num_preemptions == 1
    assert b.num_computed_tokens == 0 and not b.block_table
    sched.postprocess(seqs, [7], prefill)
    tick(sched)
    assert b.is_finished and b.completion_token_ids == [7, 7]


def test_priority_waiting_order_and_fcfs_ties():
    sched = scheduler(budget=1, scheduling_policy='priority')
    a = Sequence([1], SamplingParams(max_tokens=1), priority=4)
    b = Sequence([2], SamplingParams(max_tokens=1), priority=-1)
    c = Sequence([3], SamplingParams(max_tokens=1), priority=-1)
    for seq in (a, b, c):
        sched.add(seq)
    assert tick(sched)[0] == [b]
    assert tick(sched)[0] == [c]
    assert tick(sched)[0] == [a]


def test_empty_engine_step_does_not_dispatch_a_model_batch():
    from nanovllm.engine.llm_engine import LLMEngine
    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler()
    engine.model_runner = object()  # any attempted dispatch is an error
    assert engine.step(stream=True) == ([], 0)


def test_engine_priority_reaches_policy_and_invalid_priority_is_atomic():
    from nanovllm.engine.llm_engine import LLMEngine
    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler(budget=1, scheduling_policy='priority')
    engine.tokenizer = SimpleNamespace()
    low = engine.add_request([1], SamplingParams(max_tokens=1), priority=4)
    high = engine.add_request([2], SamplingParams(max_tokens=1), priority=-2)
    assert engine.scheduler.waiting[0].seq_id == high
    assert engine.scheduler.waiting[1].seq_id == low
    before = list(engine.scheduler.waiting)
    with pytest.raises(ValueError, match='priority'):
        engine.add_request([3], SamplingParams(max_tokens=1), priority=True)
    assert list(engine.scheduler.waiting) == before


@pytest.mark.parametrize('options', [
    {'scheduling_policy':'unknown'}, {'long_prefill_token_threshold':-1},
    {'long_prefill_token_threshold':True}, {'max_num_batched_tokens':0},
    {'max_num_seqs':0}, {'scheduler_reserve_full_isl':'false'},
    {'long_prefill_token_threshold_adaptive':1},
    {'max_model_len':8, 'long_prefill_token_threshold':9},
])
def test_invalid_scheduler_options_are_rejected(options):
    with pytest.raises(ValueError):
        scheduler(**options)


def test_config_exposes_and_validates_scheduler_options(monkeypatch, tmp_path):
    import torch
    from nanovllm.config import Config, AutoConfig
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda _: SimpleNamespace(
        dtype=torch.float16, max_position_embeddings=256))
    config = Config(str(tmp_path), scheduling_policy='priority', long_prefill_token_threshold=128,
        long_prefill_token_threshold_adaptive=True, scheduler_reserve_full_isl=False)
    assert config.scheduling_policy == 'priority' and config.max_model_len == 256
    with pytest.raises(ValueError, match='max_model_len'):
        Config(str(tmp_path), long_prefill_token_threshold=512)


def test_abort_preserves_active_shared_prefix_until_last_user_finishes():
    sched = scheduler(budget=300, blocks=3)
    prime = Sequence([5] * 256 + [6], SamplingParams(max_tokens=1))
    sched.add(prime)
    tick(sched)
    a = Sequence([5] * 256 + [8], SamplingParams(max_tokens=3))
    b = Sequence([5] * 256 + [9], SamplingParams(max_tokens=3))
    sched.add(a)
    sched.add(b)
    tick(sched)
    shared = a.block_table[0]
    assert shared == b.block_table[0]
    assert a.block_table[1] != b.block_table[1]
    assert sched.block_manager.blocks[shared].ref_count == 2
    assert sched.abort(a.seq_id)
    assert sched.block_manager.blocks[shared].ref_count == 1
    assert sched.block_manager.blocks[shared].token_ids == [5] * 256
    tick(sched)
    tick(sched)
    assert b.is_finished and b.completion_token_ids == [7, 7, 7]
    assert len(sched.block_manager.free_block_ids) == 3


def test_single_token_recompute_keeps_history_for_tp_workers():
    sched = scheduler(budget=2, long_prefill_token_threshold=1)
    seq = Sequence([1, 2], SamplingParams(max_tokens=4))
    seq.append_token(7)
    seq.append_token(8)
    sched.block_manager.allocate_slots(seq, 2)
    seq.num_computed_tokens = 2
    seq.status = SequenceStatus.RUNNING
    sched.running.append(seq)
    sched.add(Sequence([3] * 5, SamplingParams(max_tokens=1)))
    seqs, is_prefill = sched.schedule()
    assert seqs[0] is seq and is_prefill
    assert seq.num_scheduled_tokens == 1 and seq.is_prefill
    worker = pickle.loads(pickle.dumps(seq))
    assert worker.token_ids == [1, 2, 7, 8]
    assert worker.num_computed_tokens == 2
    assert worker[worker.num_computed_tokens:worker.num_computed_tokens+1] == [7]


def test_streaming_recompute_emits_only_new_tokens():
    from nanovllm.engine.llm_engine import LLMEngine

    class Runner:
        def call(self, method, seqs, is_prefill):
            return [7] * len(seqs)

    engine = object.__new__(LLMEngine)
    engine.tokenizer = SimpleNamespace()
    engine.model_runner = Runner()
    engine.scheduler = scheduler(budget=2)
    engine.add_request([1, 2, 3], SamplingParams(max_tokens=3))
    assert engine.step(stream=True)[0] == []
    first = engine.step(stream=True)[0]
    assert first[0]['token_ids'] == [7]
    seq = engine.scheduler.running.pop()
    engine.scheduler.preempt(seq)
    assert engine.step(stream=True)[0] == []  # replay first two input tokens
    second = engine.step(stream=True)[0]  # replay the old output, then sample a new one
    third = engine.step(stream=True)[0]
    assert second[0]['token_ids'] == third[0]['token_ids'] == [7]
    assert third[0]['finished'] and engine.is_finished()
    assert seq.completion_token_ids == [7, 7, 7]


@pytest.mark.parametrize('seed', range(12))
def test_varied_workloads_finish_without_duplicate_outputs_or_block_leaks(monkeypatch, seed):
    monkeypatch.setattr(Sequence, 'block_size', 16)
    rng = random.Random(seed)
    sched = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=rng.randint(1, 16),
        eos=99, kvcache_block_size=16, num_kvcache_blocks=4,
        scheduling_policy=rng.choice(['fcfs', 'priority']),
        long_prefill_token_threshold=rng.choice([0, 4]),
        long_prefill_token_threshold_adaptive=bool(seed % 2),
        scheduler_reserve_full_isl=bool(seed % 3)))
    requests = [Sequence([rng.randint(1, 3)] * rng.randint(1, 40),
        SamplingParams(max_tokens=rng.randint(1, 4)), priority=rng.randint(-2, 2)) for _ in range(6)]
    for seq in requests:
        sched.add(seq)
    for _ in range(2000):
        if sched.is_finished():
            break
        seqs, prefill = sched.schedule()
        assert sum(s.num_scheduled_tokens for s in seqs) <= sched.max_num_batched_tokens
        assert len(sched.running) <= sched.max_num_seqs
        for seq in seqs:
            end = seq.num_computed_tokens + seq.num_scheduled_tokens
            assert 0 < seq.num_scheduled_tokens <= seq.num_pending_tokens
            assert len(seq.block_table) == (end + 15) // 16
        sched.postprocess(seqs, [7] * len(seqs), prefill)
        refs = Counter(block for seq in requests for block in seq.block_table)
        for block in sched.block_manager.blocks:
            assert block.ref_count == refs[block.block_id]
        assert set(refs) == sched.block_manager.used_block_ids
        assert len(sched.block_manager.free_block_ids) + len(refs) == 4
    assert sched.is_finished(), f'workload failed to make progress: seed={seed}'
    assert all(s.completion_token_ids == [7] * s.max_tokens for s in requests)
    assert len(sched.block_manager.free_block_ids) == 4
