# vLLM 单卡 BF16 与 FP8 KV 对照

2026-11-21，单张 RTX 4090（SM 8.9），Qwen3-0.6B，vLLM 0.10.2，PyTorch 2.8.0+cu128。两组同为 V1 Triton、Eager、BF16 权重、关闭 prefix cache。固定 1536 页、每页 32 token，实测两组逻辑容量均是 49,152 token。FP8 使用 E4M3 和默认 scale，未测质量。

每轮 20 条请求，每种负载预热 2 轮、计时 5 轮，取中位输出吞吐。两组逐轮输入哈希及实际输出 token 数已核对。

| 负载 | 输入 / 固定输出 | BF16 token/s | FP8 token/s | 变化 |
|---|---|---:|---:|---:|
| 混合 | 80% 128、20% 2048 / 64 | 1521.99 | 1293.65 | −15.0% |
| 长输出 | 512 / 256 | 1906.88 | 1619.59 | −15.1% |

在同逻辑容量下，FP8 KV 数据载荷估算由 5.25 GiB 降至 2.625 GiB，模型权重、scale、运行工作区另外计算。此结果仅对应测试的后端和负载。尚未测同显存预算下可承载更高并发的收益，也未做 FlashInfer 等后端对照、精度对照或算子 profile。

[BF16 数据](raw/auto.json) · [FP8 数据](raw/fp8_e4m3.json) · [复现脚本](raw/measure.py)

复现时使用 vLLM 专用环境，设置 `CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_V1=1`。分别运行 `python measure.py --kv auto --out auto.json` 与 `python measure.py --kv fp8_e4m3 --out fp8_e4m3.json`。脚本中的模型路径按本机位置填写。

环境验证记录：启用动态 scale 时，vLLM 0.10.2 在预热阶段遇到缺失 attention metadata。关闭动态 scale 后，16 token 页也被 Triton FP8 路径拒绝。最终对照两组均使用 32 token 页。[动态 scale 启动日志](raw/fp8-dynamic-scales-init-failure.txt) · [16 token 页错误](raw/fp8-block16-failure.txt)
