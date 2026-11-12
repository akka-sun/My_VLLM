from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from safetensors.torch import save_file

from nanovllm.layers.linear import ReplicatedLinear, ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear, QKVParallelLinear
from nanovllm.utils.loader import load_model


def tp(monkeypatch, rank=0, size=1):
    monkeypatch.setattr(dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(dist, 'get_world_size', lambda: size)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_int8_storage_zero_rows_and_numeric_error(monkeypatch, dtype):
    tp(monkeypatch)
    layer = ReplicatedLinear(64, 16, bias=True, quantization='int8')
    torch.manual_seed(5)
    weight = torch.randn(16, 64)
    weight[0].zero_()
    bias = torch.randn(16).to(dtype)
    layer.weight_loader(layer.weight, weight)
    layer.weight_loader(layer.bias, bias)
    x = torch.randn(7, 64).to(dtype)
    actual = layer(x)
    expected = torch.nn.functional.linear(x.float(), weight, bias.float())
    assert layer.weight.dtype == torch.int8 and not layer.weight.requires_grad
    assert layer.weight_scale.dtype == torch.float32
    assert torch.isfinite(layer.weight_scale).all()
    assert torch.equal(layer.weight[0], torch.zeros(64, dtype=torch.int8))
    assert actual.dtype == dtype
    # Error budget includes per-channel quantization and activation precision.
    assert (actual.float() - expected).abs().mean() < 0.07
    assert layer.weight.numel() * layer.weight.element_size() + layer.weight_scale.numel() * 4 < weight.numel() * 2


def test_column_and_row_parallel_quantize_after_sharding(monkeypatch):
    tp(monkeypatch, rank=1, size=2)
    weight = torch.tensor([[1., 2., 30., 40.], [-4., 8., -90., 10.], [5., 6., 70., 80.], [9., 10., 110., 120.]])
    column = ColumnParallelLinear(4, 4, quantization='int8')
    row = RowParallelLinear(4, 4, bias=True, quantization='int8')
    column.weight_loader(column.weight, weight)
    row.weight_loader(row.weight, weight)
    row.weight_loader(row.bias, torch.ones(4))
    torch.testing.assert_close(column.weight.float() * column.weight_scale, weight[2:], atol=0.5, rtol=0)
    torch.testing.assert_close(row.weight.float() * row.weight_scale, weight[:, 2:], atol=0.5, rtol=0)
    # TP row bias must be added only on rank zero, before the sum.
    monkeypatch.setattr(dist, 'all_reduce', lambda y: None)
    x = torch.tensor([[1., 2.]], dtype=torch.float16)
    torch.testing.assert_close(row(x), torch.nn.functional.linear(x, (row.weight.float() * row.weight_scale).half()), atol=0.1, rtol=0)


def test_packed_qkv_and_gate_up_scales_follow_output_offsets(monkeypatch):
    tp(monkeypatch, rank=1, size=2)
    qkv = QKVParallelLinear(3, 1, 2, 2, quantization='int8')
    merged = MergedColumnParallelLinear(3, [4, 2], quantization='int8')
    values = [torch.full((2, 3), float(n)) for n in [1, 20, 90]]
    for shard, weight in zip(['q', 'k', 'v'], values):
        qkv.weight_loader(qkv.weight, weight, shard)
    expected = torch.tensor([[1.] * 3, [20.] * 3, [90.] * 3])
    torch.testing.assert_close(qkv.weight.float() * qkv.weight_scale, expected)
    merged.weight_loader(merged.weight, torch.full((4, 3), 7.), 0)
    merged.weight_loader(merged.weight, torch.full((2, 3), 70.), 1)
    torch.testing.assert_close(merged.weight.float() * merged.weight_scale,
        torch.tensor([[7.] * 3, [7.] * 3, [70.] * 3]))


def test_gqa_distinct_rows_and_row_parallel_sum(monkeypatch):
    q = torch.arange(1, 25, dtype=torch.float32).reshape(8, 3)
    k = torch.arange(31, 43, dtype=torch.float32).reshape(4, 3)
    v = torch.arange(51, 63, dtype=torch.float32).reshape(4, 3)
    weight = torch.tensor([[1., 2., 3., 4.], [-4., 3., 2., 1.]])
    x = torch.tensor([[2., -1., 3., 2.]], dtype=torch.float16)
    partials = []
    monkeypatch.setattr(dist, 'all_reduce', lambda y: None)
    for rank in range(2):
        tp(monkeypatch, rank, 2)
        layer = QKVParallelLinear(3, 2, 4, 2, quantization='int8')
        for shard, values in [('q', q), ('k', k), ('v', v)]:
            layer.weight_loader(layer.weight, values, shard)
        expected = torch.cat([q[rank*4:rank*4+4], k[rank*2:rank*2+2], v[rank*2:rank*2+2]])
        torch.testing.assert_close(layer.weight.float() * layer.weight_scale, expected, atol=0.25, rtol=0)
        row = RowParallelLinear(4, 2, bias=True, quantization='int8')
        row.weight_loader(row.weight, weight)
        row.weight_loader(row.bias, torch.tensor([1., 2.]))
        partials.append(row(x[:, rank*2:rank*2+2]))
    expected = torch.nn.functional.linear(x, weight.half(), torch.tensor([1., 2.]).half())
    torch.testing.assert_close(sum(partials), expected, atol=0.1, rtol=0)


def test_qwen_construction_keeps_embedding_head_float_and_quantizes_all_projections(monkeypatch):
    import sys
    import types
    import importlib.util
    from pathlib import Path
    tp(monkeypatch)

    class UnavailableKernel(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    for path, name in [('activation', 'SiluAndMul'), ('attention', 'Attention'), ('layernorm', 'RMSNorm')]:
        module = types.ModuleType('nanovllm.layers.' + path)
        setattr(module, name, UnavailableKernel)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    rope = types.ModuleType('nanovllm.layers.rotary_embedding')
    rope.get_rope = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, rope.__name__, rope)
    spec = importlib.util.spec_from_file_location('quant_qwen', Path(__file__).resolve().parents[1] / 'nanovllm/models/qwen3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(vocab_size=32, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rms_norm_eps=1e-6, attention_bias=False, head_dim=4, intermediate_size=32,
        hidden_act='silu', tie_word_embeddings=True)
    model = module.Qwen3ForCausalLM(config, quantization='int8')
    from nanovllm.layers.linear import LinearBase
    projections = [m for m in model.modules() if isinstance(m, LinearBase)]
    assert len(projections) == 8 and all(m.weight.dtype == torch.int8 for m in projections)
    assert model.model.embed_tokens.weight.is_floating_point()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_safetensors_loader_quantizes_packed_weights_and_preserves_bias(monkeypatch, tmp_path):
    tp(monkeypatch)
    model = nn.Module()
    model.proj = MergedColumnParallelLinear(3, [2, 2], bias=True, quantization='int8')
    model.packed_modules_mapping = {'gate_proj': ('proj', 0), 'up_proj': ('proj', 1)}
    save_file({'gate_proj.weight': torch.ones(2, 3), 'up_proj.weight': torch.full((2, 3), 20.),
        'gate_proj.bias': torch.full((2,), 3.), 'up_proj.bias': torch.full((2,), 4.)}, str(tmp_path / 'model.safetensors'))
    load_model(model, str(tmp_path))
    torch.testing.assert_close(model.proj.weight.float() * model.proj.weight_scale,
        torch.tensor([[1.] * 3, [1.] * 3, [20.] * 3, [20.] * 3]))
    assert model.proj.bias.tolist() == [3., 3., 4., 4.]


def test_loader_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(nn.Module(), str(tmp_path))


def test_invalid_quantization_mode_rejected(monkeypatch):
    tp(monkeypatch)
    with pytest.raises(ValueError, match='quantization'):
        ReplicatedLinear(2, 2, quantization='int4')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(5, 33, 80), (1, 129, 2048), (33, 64, 256)])
def test_fused_cuda_int8_linear_matches_reference_and_graph(monkeypatch, dtype, shape):
    pytest.importorskip('triton')
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    tp(monkeypatch)
    m, n, k = shape
    layer = ReplicatedLinear(k, n, bias=True, quantization='int8').cuda()
    torch.manual_seed(9)
    layer.weight_loader(layer.weight, torch.randn(n, k) * 0.1)
    layer.weight_loader(layer.bias, torch.randn(n))
    x = torch.randn(m, k, device='cuda', dtype=dtype)
    weight = (layer.weight.float() * layer.weight_scale).to(dtype)
    expected = torch.nn.functional.linear(x, weight, layer.bias.to(dtype))
    actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    # All weights/scales must stay on-device; no temporary persistent FP16 copy.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(x)
    graph.replay()
    torch.testing.assert_close(output, expected, atol=0.03, rtol=0.03)
    x.mul_(2)
    graph.replay()
    torch.testing.assert_close(output, torch.nn.functional.linear(x, weight, layer.bias.to(dtype)), atol=0.06, rtol=0.03)
