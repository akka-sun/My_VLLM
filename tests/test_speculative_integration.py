"""Exercise the native target, scheduler and both proposers with tiny weights.

Only unavailable GPU attention kernels and host-to-CUDA transfers are replaced;
Qwen3 weights, projections, feature capture and sampling are production code.
"""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context


@pytest.fixture
def native_models(monkeypatch, tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from nanovllm.layers.layernorm import RMSNorm
    from nanovllm.layers.activation import SiluAndMul
    from nanovllm.layers.rotary_embedding import RotaryEmbedding, get_rope
    from nanovllm.utils.loader import load_model
    monkeypatch.setattr(Sequence, 'block_size', 4)
    for cls, names in ((RMSNorm, ['rms_forward', 'add_rms_forward']),
                       (SiluAndMul, ['forward']), (RotaryEmbedding, ['forward'])):
        for name in names:
            fn = getattr(cls, name)
            monkeypatch.setattr(cls, name, getattr(fn, '__wrapped__', fn))
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    class ReferencePagedAttention(nn.Module):
        def __init__(self, heads, dim, scale, kv_heads):
            super().__init__()
            self.heads, self.kv_heads = heads, kv_heads
            self.k_cache = torch.zeros(16, 4, kv_heads, dim, dtype=torch.float64)
            self.v_cache = torch.zeros_like(self.k_cache)

        def forward(self, q, k, v):
            ctx = get_context()
            self.k_cache.flatten(0, 1)[ctx.slot_mapping.long()] = k
            self.v_cache.flatten(0, 1)[ctx.slot_mapping.long()] = v
            outputs = []
            for i in range(len(ctx.cu_seqlens_q) - 1):
                start, end = ctx.cu_seqlens_q[i:i+2].tolist()
                length = int(ctx.cu_seqlens_k[i+1] - ctx.cu_seqlens_k[i])
                if ctx.block_tables is None:
                    ks, ke = ctx.cu_seqlens_k[i:i+2].tolist()
                    key, value = k[ks:ke], v[ks:ke]
                else:
                    pages = ctx.block_tables[i]
                    pages = pages[pages >= 0].long()
                    key = self.k_cache[pages].flatten(0, 1)[:length]
                    value = self.v_cache[pages].flatten(0, 1)[:length]
                repeat = self.heads // self.kv_heads
                key, value = key.repeat_interleave(repeat, 1), value.repeat_interleave(repeat, 1)
                mask = torch.arange(length)[None, :] <= torch.arange(length - (end-start), length)[:, None]
                out = F.scaled_dot_product_attention(q[start:end].transpose(0, 1),
                    key.transpose(0, 1), value.transpose(0, 1), attn_mask=mask)
                outputs.append(out.transpose(0, 1))
            return torch.cat(outputs)

    attention = types.ModuleType('nanovllm.layers.attention')
    attention.Attention = ReferencePagedAttention
    monkeypatch.setitem(sys.modules, 'nanovllm.layers.attention', attention)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('native_spec_qwen3', root / 'nanovllm/models/qwen3.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    monkeypatch.setitem(sys.modules, 'nanovllm.models.qwen3', native)
    spec = importlib.util.spec_from_file_location('native_spec_runner', root / 'nanovllm/engine/model_runner.py')
    execution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(execution)
    tensor = torch.tensor
    monkeypatch.setattr(torch, 'tensor', lambda *a, **kw: tensor(*a, **{k:v for k,v in kw.items() if k != 'pin_memory'}))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kwargs: self)
    torch.manual_seed(24)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=64, attention_bias=False)
    reference = Qwen3ForCausalLM(config).double().eval()
    reference.save_pretrained(tmp_path)
    target = native.Qwen3ForCausalLM(config).double().eval()
    load_model(target, str(tmp_path))
    runner = object.__new__(execution.ModelRunner)
    runner.model, runner.rank, runner.block_size, runner.world_size = target, 0, 4, 1
    yield config, target, reference, runner
    reset_context()
    get_rope.cache_clear()


@pytest.mark.parametrize('method', ['draft_model', 'dspark'])
def test_native_target_speculation_matches_greedy_reference_and_streams(native_models, method):
    from nanovllm.engine.drafters import IndependentDrafter, DSparkDrafter
    from nanovllm.models.dspark import DSparkModel
    config, target, reference, runner = native_models
    options = dict(method=method, model='unused', num_speculative_tokens=3)
    engine_config = SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=8,
        max_model_len=64, eos=-1, kvcache_block_size=4, num_kvcache_blocks=4,
        long_prefill_token_threshold=2, speculative_config=options)
    if method == 'draft_model':
        runner.drafter = IndependentDrafter(reference)
    else:
        dc = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=16,
            markov_rank=4, target_layer_ids=[0, 1], mask_token_id=0,
            enable_confidence_head=True, rms_norm_eps=1e-6)
        drafter_model = DSparkModel(dc, 16).double().eval()
        runner.drafter = DSparkDrafter(drafter_model, engine_config)
        target.model.aux_layer_ids = (0, 1)
    engine = object.__new__(LLMEngine)
    engine.config, engine.model_runner = engine_config, runner
    engine.tokenizer = SimpleNamespace()
    engine.scheduler = Scheduler(engine_config)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
    expected = []
    for prompt in prompts:
        ids = prompt.copy()
        with torch.inference_mode():
            for _ in range(8):
                ids.append(reference(torch.tensor([ids])).logits[0, -1].argmax().item())
        expected.append(ids[len(prompt):])
    requests = [engine.add_request(p, SamplingParams(temperature=0, max_tokens=8)) for p in prompts]
    output = {key: [] for key in requests}
    rounds = 0
    delta_lengths = []
    while not engine.is_finished():
        deltas, work = engine.step(stream=True)
        for delta in deltas:
            output[delta['seq_id']].extend(delta['token_ids'])
            delta_lengths.append(len(delta['token_ids']))
        rounds += 1
        assert rounds < 40
    assert list(output.values()) == expected
    assert len(engine.scheduler.block_manager.free_block_ids) == 4
    assert engine.scheduler.num_preemptions > 0
    state = runner.drafter.caches if method == 'draft_model' else runner.drafter.slots
    assert not state
    if method == 'draft_model':
        # Identical target/draft weights accept complete blocks.
        assert max(delta_lengths) > 1
