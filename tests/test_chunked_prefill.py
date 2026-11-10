from types import SimpleNamespace
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import pytest
import torch

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context


def scheduler(budget=4, slots=4, blocks=16):
    return Scheduler(SimpleNamespace(max_num_seqs=slots, max_num_batched_tokens=budget,
        eos=99, kvcache_block_size=256, num_kvcache_blocks=blocks))


def tick(sched):
    seqs, prefill = sched.schedule()
    counts = [s.num_scheduled_tokens for s in seqs]
    sched.postprocess(seqs, [7] * len(seqs), prefill)
    return seqs, counts, prefill


def test_decode_and_long_prompt_share_each_step_budget():
    sched = scheduler()
    short = Sequence([1], SamplingParams(max_tokens=4))
    sched.add(short)
    tick(sched)
    long = Sequence(list(range(8)), SamplingParams(max_tokens=1))
    sched.add(long)
    seqs, counts, prefill = tick(sched)
    assert seqs == [short, long]
    assert counts == [1, 3] and prefill
    assert short.num_completion_tokens == 2
    assert long.num_cached_tokens == 3 and long.num_completion_tokens == 0
    tick(sched)
    assert short.num_completion_tokens == 3 and long.num_cached_tokens == 6
    tick(sched)
    assert short.is_finished and long.is_finished
    assert sched.is_finished()
    assert len(sched.block_manager.free_block_ids) == 16


def test_token_budget_also_limits_decode_only_batches():
    sched = scheduler(budget=4)
    seqs = [Sequence([i], SamplingParams(max_tokens=2)) for i in range(4)]
    for seq in seqs:
        sched.add(seq)
    tick(sched)
    sched.max_num_batched_tokens = 2
    _, counts, prefill = tick(sched)
    assert counts == [1, 1] and not prefill
    tick(sched)
    assert sched.is_finished()


def test_multiple_prompts_can_use_partial_remaining_budget():
    sched = scheduler(budget=4)
    a, b = Sequence([1]), Sequence([2] * 7)
    sched.add(a)
    sched.add(b)
    seqs, counts, _ = tick(sched)
    assert seqs == [a, b] and counts == [1, 3]
    assert b.num_completion_tokens == 0


def test_prefix_cache_reuse_resumes_at_cached_block():
    sched = scheduler(budget=300)
    tokens = list(range(260))
    first = Sequence(tokens, SamplingParams(max_tokens=1))
    sched.add(first)
    tick(sched)
    second = Sequence(tokens, SamplingParams(max_tokens=1))
    sched.add(second)
    seqs, prefill = sched.schedule()
    assert second.num_cached_tokens == 256
    assert second.num_scheduled_tokens == 4
    sched.postprocess(seqs, [7], prefill)
    assert second.is_finished


def test_cache_pressure_preempts_and_recomputes_without_duplicate_output():
    sched = scheduler(budget=512, blocks=2)
    a = Sequence([1] * 256, SamplingParams(max_tokens=2))
    b = Sequence([2] * 256, SamplingParams(max_tokens=2))
    sched.add(a)
    sched.add(b)
    tick(sched)
    tick(sched)
    assert a.is_finished and b.num_completion_tokens == 1
    tick(sched)
    assert b.is_finished and b.completion_token_ids == [7, 7]


def test_incremental_partial_prompt_cache_preserves_decode_progress_under_pressure():
    sched = scheduler(budget=257, blocks=3)
    a = Sequence([1] * 256, SamplingParams(max_tokens=2))
    b = Sequence([2] * 257, SamplingParams(max_tokens=1))
    sched.add(a)
    sched.add(b)
    tick(sched)  # A completes prefill; B only allocates its first page.
    assert b.num_cached_tokens == 1
    assert len(sched.block_manager.free_block_ids) == 1
    tick(sched)  # A can acquire the remaining page and finish its decode.
    assert a.is_finished
    for _ in range(3):
        if sched.is_finished():
            break
        tick(sched)
    assert b.is_finished
    assert a.completion_token_ids == [7, 7] and b.completion_token_ids == [7]


def load_runner(monkeypatch):
    model = types.ModuleType('nanovllm.models.qwen3')
    model.Qwen3ForCausalLM = object
    monkeypatch.setitem(sys.modules, 'nanovllm.models.qwen3', model)
    path = Path(__file__).resolve().parents[1] / 'nanovllm/engine/model_runner.py'
    spec = importlib.util.spec_from_file_location('chunk_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelRunner


def test_mixed_tp_batch_builds_positions_slots_and_full_context(monkeypatch):
    runner = object.__new__(load_runner(monkeypatch))
    runner.block_size = 256
    decode = Sequence([1, 2])
    decode.num_cached_tokens = 2
    decode.append_token(7)
    decode.is_prefill = False
    decode.num_scheduled_tokens = 1
    decode.block_table = [3]
    # Decode workers only receive the final token, not the full token list.
    decode = pickle.loads(pickle.dumps(decode))
    prompt = Sequence([4, 5, 6, 8, 9])
    prompt.num_cached_tokens = 2
    prompt.num_scheduled_tokens = 2
    prompt.block_table = [1]
    tensor = torch.tensor
    monkeypatch.setattr(torch, 'tensor', lambda *a, **kw: tensor(*a, **{k:v for k,v in kw.items() if k != 'pin_memory'}))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kw: self)
    try:
        ids, positions = runner.prepare_prefill([decode, prompt])
        ctx = get_context()
        assert ids.tolist() == [7, 6, 8]
        assert positions.tolist() == [2, 2, 3]
        assert ctx.slot_mapping.tolist() == [770, 258, 259]
        assert ctx.cu_seqlens_q.tolist() == [0, 1, 3]
        assert ctx.cu_seqlens_k.tolist() == [0, 3, 7]
        assert ctx.block_tables.tolist() == [[3], [1]]
    finally:
        reset_context()


def test_chunk_attention_matches_full_causal_attention_with_paged_prefix(monkeypatch):
    """Run production Attention.forward; replace only unavailable GPU kernels."""
    triton = types.ModuleType('triton')
    triton.jit = lambda fn: fn
    language = types.ModuleType('triton.language')
    language.constexpr = object()
    flash = types.ModuleType('flash_attn')

    def varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, block_table, **kwargs):
        outputs = []
        for i in range(len(cu_seqlens_q) - 1):
            qs, qe = cu_seqlens_q[i:i+2].tolist()
            ks, ke = cu_seqlens_k[i:i+2].tolist()
            n = ke - ks
            if block_table is not None:
                pages = block_table[i][block_table[i] >= 0].long()
                ki, vi = k[pages].flatten(0, 1)[:n], v[pages].flatten(0, 1)[:n]
            else:
                ki, vi = k[ks:ke], v[ks:ke]
            qi = q[qs:qe].transpose(0, 1)
            scores = qi @ ki.transpose(0, 1).transpose(1, 2) * kwargs['softmax_scale']
            mask = torch.arange(n)[None, :] <= torch.arange(n - (qe - qs), n)[:, None]
            outputs.append((scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vi.transpose(0, 1)).transpose(0, 1))
        return torch.cat(outputs)

    flash.flash_attn_varlen_func = varlen
    flash.flash_attn_with_kvcache = lambda *a, **kw: pytest.fail('mixed batch must use varlen attention')
    for name, module in [('triton', triton), ('triton.language', language), ('flash_attn', flash)]:
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('chunk_attention', Path(__file__).resolve().parents[1] / 'nanovllm/layers/attention.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def store(k, v, kc, vc, slots):
        kc.flatten(0, 1)[slots.long()] = k
        vc.flatten(0, 1)[slots.long()] = v

    monkeypatch.setattr(module, 'store_kvcache', store)
    torch.manual_seed(12)
    q, k, v = [torch.randn(6, 1, 4) for _ in range(3)]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True).transpose(0, 1)
    attention = module.Attention(1, 4, 0.5, 1)
    attention.k_cache = torch.zeros(2, 4, 1, 4)
    attention.v_cache = torch.zeros(2, 4, 1, 4)
    # Logical first page lives in physical page 1; second in page 0.
    outputs = []
    try:
        for start, end in [(0, 2), (2, 5), (5, 6)]:
            slots = torch.tensor([4 + i if i < 4 else i - 4 for i in range(start, end)])
            set_context(True, torch.tensor([0, end-start]), torch.tensor([0, end]),
                end-start, end, slots, None, torch.tensor([[1, 0]]))
            outputs.append(attention(q[start:end], k[start:end], v[start:end]))
        torch.testing.assert_close(torch.cat(outputs), expected)
    finally:
        reset_context()
