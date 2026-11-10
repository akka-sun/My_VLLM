"""Optional integration test using real Triton and FlashAttention kernels."""
import importlib.util

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_real_paged_chunk_attention_matches_full_attention():
    if any(importlib.util.find_spec(name) is None for name in ('triton', 'flash_attn')):
        pytest.skip('Triton/FlashAttention unavailable')
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import set_context, reset_context

    torch.manual_seed(17)
    q, k, v = [torch.randn(6, 2, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True).transpose(0, 1)
    attention = Attention(2, 64, 0.125, 2)
    attention.k_cache = torch.zeros(2, 256, 2, 64, device='cuda', dtype=torch.float16)
    attention.v_cache = torch.zeros_like(attention.k_cache)
    outputs = []
    try:
        for start, end in [(0, 2), (2, 5), (5, 6)]:
            set_context(True,
                torch.tensor([0, end-start], dtype=torch.int32, device='cuda'),
                torch.tensor([0, end], dtype=torch.int32, device='cuda'),
                end-start, end,
                torch.arange(256+start, 256+end, dtype=torch.int32, device='cuda'),
                None, torch.tensor([[1]], dtype=torch.int32, device='cuda'))
            outputs.append(attention(q[start:end], k[start:end], v[start:end]))
        torch.testing.assert_close(torch.cat(outputs), expected, atol=3e-3, rtol=3e-3)
    finally:
        reset_context()
