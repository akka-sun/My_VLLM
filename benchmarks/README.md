# 在线评测（第一版）

对已启动的 OpenAI 兼容 chat SSE 服务做闭环并发测量。JSONL 每行一个请求，例如：

```json
{"id":"1","messages":[{"role":"user","content":"你好"}]}
{"id":"2","messages":[{"role":"user","content":[{"type":"text","text":"图中有什么？"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]}
```

图像URL必须是服务端可以访问的固定内容；为可复现，建议使用固定内容的data URL并冻结JSONL的哈希。不要将私有图片或密钥提交仓库。目前Nano-vLLM的API只接受文本，图像示例要求支持图像的服务。

```bash
python -m benchmarks.serve --url http://127.0.0.1:8000/v1/chat/completions --model nano-vllm --workload workload.jsonl --output runs/baseline --concurrency 4
python -m benchmarks.serve --url http://127.0.0.1:8001/v1/chat/completions --model nano-vllm --workload workload.jsonl --output runs/optimized --concurrency 4
python -m benchmarks.compare runs/baseline/metrics.json runs/optimized/metrics.json
```

输出每请求结果和metrics.json，后者包含工作负载文件SHA256、模型名、并发、输出上限、客户端环境和端点。比较器要求这些配置与工作负载一致；须自行记录服务端模型revision、GPU预算、版本、精度、优化开关及图片内容，确认一致后才可归因。若服务未返回SSE usage，输出吞吐为null；失败请求单独计数。TTFT是从HTTP发送到首个非空文本chunk的时间，P95使用nearest-rank；不是首token内部就绪时间。没有输出内容或[DONE]视为失败。采样为temperature=0，服务实现仍可能存在非确定性。

这是基础版：不生成工作负载、不测质量、显存、开环到达或逐token ITL，也不自动验证服务端配置和动态图片内容。为避免延迟统计掩盖失败，比较报告需同时查看成功/失败数。性能收益需在相同硬件与工作量下多次实测；本项目当前尚无多模态、PD或KV量化服务。

## CPU 环境与固定工作负载

在已安装 PyTorch 的 AutoDL 环境中，从仓库根目录运行：

```bash
bash scripts/setup_cpu.sh
source .venv/bin/activate
TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m scripts.prepare_models --output ../models
python -m benchmarks.workloads --preset mixed --count 200 --seed 42 --output ../runs/mixed.jsonl
```

CPU 测试默认复用平台 PyTorch。内存受限时可运行 `bash scripts/setup_cpu.sh --cpu-wheels`，在 venv 内安装 PyTorch 和 torchvision 的 CPU 版本。后续 CUDA 推理使用单独的 CUDA 环境。模型下载固定到具体 Hugging Face commit，并在各模型目录写入 `preparation.json`。可以用 `--model` 选择仓库、`--revision` 重现指定版本、`--metadata-only` 仅下载配置和 tokenizer。使用不同输出目录保存同一模型的不同版本。

工作负载生成器输出可直接交给 `benchmarks.serve` 的文本 JSONL。同样的 seed、preset 和 count 会生成相同的文件。混合负载约五分之一为长文本，长度按模板重复数定义。实际输入 token 数以目标 tokenizer 和 chat template 为准。


压测可用 `--warmup 10` 在计时前顺序发送 10 条请求。预热从固定工作负载取样，会改变前缀缓存状态。冷缓存实验可保持默认值 0，并确保服务缓存处在所需状态。`--server-metadata server.json` 可把模型 revision、服务端 commit、GPU 型号与数量、精度及优化开关随客户端参数保存到 `manifest.json`。这些字段来自调用方提供的文件，仍需核对服务端实际部署。
