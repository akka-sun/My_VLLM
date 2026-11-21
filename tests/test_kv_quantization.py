import torch

from nanovllm.layers.kv_quantization import packed_cache_slots, read_int8_cache, write_int8_cache


def test_int8_cache_gathers_only_logical_slots_and_preserves_zero():
    table = torch.tensor([[2, 0], [1, -1]], dtype=torch.int32)
    lengths = torch.tensor([0, 3, 4], dtype=torch.int32)
    slots = packed_cache_slots(table, lengths, block_size=2, total_tokens=4)
    assert slots.tolist() == [4, 5, 0, 2]

    values = torch.tensor([[[0., 0.]], [[.2, 1.]], [[-20., 9.]], [[2., 5.]]])
    cache = torch.empty(3, 2, 1, 2, dtype=torch.int8)
    scales = torch.empty(3, 2, 1, 1)
    write_int8_cache(values, cache, scales, slots)
    restored = read_int8_cache(cache, scales, slots, torch.float32)
    assert torch.equal(restored[0], values[0])
    assert torch.all((restored - values).abs() <= values.abs().amax(dim=-1, keepdim=True) / 127)


def test_cuda_int8_decode_reads_ragged_paged_history():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('flash_attn')
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import reset_context, set_context

    torch.manual_seed(7)
    device = 'cuda'
    q = torch.rand(2, 2, 32, device=device, dtype=torch.float16)
    k = torch.rand(2, 1, 32, device=device, dtype=torch.float16)
    v = torch.rand_like(k)
    float_attention = Attention(2, 32, 32 ** -.5, 1)
    quant_attention = Attention(2, 32, 32 ** -.5, 1)
    pages = [torch.rand(3, 256, 1, 32, device=device, dtype=q.dtype) for _ in range(2)]
    float_attention.k_cache, float_attention.v_cache = pages
    quant_attention.k_cache, quant_attention.v_cache = [torch.empty_like(page, dtype=torch.int8) for page in pages]
    quant_attention.k_scale, quant_attention.v_scale = [
        torch.empty(3, 256, 1, 1, device=device, dtype=torch.float32) for _ in pages]
    all_slots = torch.arange(3 * 256, device=device)
    for page, cache, scales in zip(
            pages, (quant_attention.k_cache, quant_attention.v_cache),
            (quant_attention.k_scale, quant_attention.v_scale)):
        write_int8_cache(page.flatten(0, 1), cache, scales, all_slots)
    table = torch.tensor([[2, 0], [1, -1]], device=device, dtype=torch.int32)
    lengths = torch.tensor([259, 9], device=device, dtype=torch.int32)
    new_slots = torch.tensor([2, 264], device=device, dtype=torch.int32)
    try:
        set_context(False, slot_mapping=new_slots, context_lens=lengths, block_tables=table)
        expected = float_attention(q, k, v).view_as(q)
        set_context(False, slot_mapping=new_slots, context_lens=lengths, block_tables=table)
        actual = quant_attention(q, k, v).view_as(q)
        torch.testing.assert_close(actual, expected, atol=.004, rtol=.004)
    finally:
        reset_context()


def test_selective_layers_validate_and_preserve_explicit_order(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest
    from nanovllm.config import AutoConfig, Config

    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda _: SimpleNamespace(
        dtype=torch.bfloat16, max_position_embeddings=4096, num_hidden_layers=4))
    options = dict(enforce_eager=True, kv_cache_dtype='int8-reference')
    config = Config(str(tmp_path), kv_int8_layers=[3, 1], **options)
    assert config.kv_int8_layers == (1, 3)
    for invalid in ([], [True], [2, 2], [-1], [4]):
        with pytest.raises(ValueError, match='kv_int8_layers'):
            Config(str(tmp_path), kv_int8_layers=invalid, **options)
    with pytest.raises(ValueError, match='kv_int8_layers'):
        Config(str(tmp_path), kv_int8_layers=[1])
