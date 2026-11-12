"""Fused INT8-weight / 16-bit-activation matmul, without an FP16 weight copy."""
import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, W, S, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    scales = tl.load(S + cols, cols < N, other=0)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        offsets = block * BK + ks
        x = tl.load(X + rows[:, None] * K + offsets[None, :],
                    (rows[:, None] < M) & (offsets[None, :] < K), other=0)
        w = tl.load(W + cols[None, :] * K + offsets[:, None],
                    (cols[None, :] < N) & (offsets[:, None] < K), other=0)
        # Dequantize only the current tile, matching the activation dtype.
        w = (w.to(tl.float32) * scales[None, :]).to(x.dtype)
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def linear(x, weight, scale, bias=None):
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x.shape
    n = weight.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m:
        _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            x, weight, scale, bias, y, m, n, k, bias is not None, 16, 64, 32)
    return y.view(*shape, n)
