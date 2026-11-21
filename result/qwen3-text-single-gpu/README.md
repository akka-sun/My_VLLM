# Qwen3-0.6B 单卡文本基线与优化

2026-11-21，RTX 4090 单卡，PyTorch 2.8.0+cu128，FlashAttention 2.8.3。模型 revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`。所有对照统一关闭 TorchDynamo。测量发生在相关改动提交前，原始 manifest 保留当时的 HEAD，最终实现提交为 `20cc4a9`。

离线负载有 20 条请求，其中 20% 输入为 2048 token，其余为 128 token。每条固定输出 64 token，预热 8 条，重复 3 次。各配置的逐轮负载哈希一致。

| 模式 | 中位输出 token/s | 相对基线 |
|---|---:|---:|
| [BF16 Eager](raw/baseline/metrics.json) | 499.80 | — |
| [CUDA Graph](raw/graph/metrics.json) | 865.51 | +73.2% |
| [Chunked Prefill，阈值 128](raw/chunked/metrics.json) | 508.30 | +1.7% |
| [W8A16 权重量化](raw/w8a16/metrics.json) | 418.64 | −16.2% |
| [INT8 KV 参考实现](raw/kv-int8-reference/metrics.json) | 362.94 | −27.4% |

W8A16 首轮有额外耗时。INT8 KV 的显存记录包含按预算分配的缓存池和工作区，不能据此推断同容量的显存收益。

在线 SSE 使用 [固定文本负载](raw/mixed.jsonl) 和 [服务端清单](raw/online-server.json)。每档 40 条请求，预热 6 条，输出上限 32 token。P95 首字时延使用客户端首个可见文本的时间。

| 并发 | 成功/失败 | 输出 token/s | P95 首字时延 | P95 端到端时延 |
|---|---:|---:|---:|---:|
| [1](raw/online-c1/metrics.json) | 40/0 | 42.32 | 145.31 ms | 820.88 ms |
| [4](raw/online-c4/metrics.json) | 40/0 | 144.19 | 171.36 ms | 912.12 ms |
| [16](raw/online-c16/metrics.json) | 40/0 | 315.26 | 636.49 ms | 1733.99 ms |
| [32](raw/online-c32/metrics.json) | 40/0 | 374.34 | 1290.50 ms | 2669.80 ms |

CUDA 回归 [134 项通过](raw/pytest.txt)。TP2/TP4 效果待多卡实测。

## 2026-11-21 补测：同页数下的选择性 KV 量化

原模型和离线负载保持一致。使用 Eager、禁用 TorchDynamo，固定为 192 页（每页 256 token）。选择性方案对中间 14 层量化，保留首尾各 7 层的 BF16。缓存池统计含 scale，反量化工作区另计。

| 模式 | 中位输出 token/s | KV 池 GiB |
|---|---:|---:|
| [全 BF16](raw/2026-11-21-kv-selective/bf16/metrics.json) | 499.27 | 5.250 |
| [全 INT8](raw/2026-11-21-kv-selective/full-int8/metrics.json) | 377.14 | 2.707 |
| [14 层 INT8](raw/2026-11-21-kv-selective/selective/metrics.json) | 427.43 | 3.979 |

该对照仅测吞吐和缓存池占用。层敏感性、任务准确率和多卡效果尚待实测。
