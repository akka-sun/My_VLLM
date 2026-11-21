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

## 单卡与多卡文本基线

在装有 CUDA 版 PyTorch 和 FlashAttention 的环境中运行。同一台主机内按顺序启动各个配置，当前引擎使用固定的 NCCL 端口 2333 和共享内存名称。

```bash
model=../models/Qwen--Qwen3-0.6B
python -m benchmarks.offline --model "$model" --devices 0 --tp 1 \
  --mode baseline --output ../runs/text/tp1-baseline
python -m benchmarks.offline --model "$model" --devices 0,1 --tp 2 \
  --mode baseline --output ../runs/text/tp2-baseline
```

改用 `--devices 0,1,2,3 --tp 4` 测四卡 TP。TP 需整除模型的 attention heads 和 KV heads。使用相同种子及长度配置，核对每轮工作负载哈希。基线使用 Eager 和原权重精度，分别指定 `--mode graph`或 `--mode chunked` 做单项对照。默认负载中约 20% 的输入为 2048 token，其余为 128 token，每请求固定输出 64 token。结果包含总输出吞吐及 rank 0 的 PyTorch 峰值分配量，后者包含 KV 池，不能当作所有 GPU 显存总和。

在线评测的 TP 由服务端控制。先启动服务，再从另一个终端用本页的 `benchmarks.serve` 命令发请求：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m nanovllm.server --model "$model" \
  --tensor-parallel-size 2 --port 8000 --enforce-eager \
  --max-num-batched-tokens 512 --max-num-seqs 32 --gpu-memory-utilization 0.35
```

单卡使用 `CUDA_VISIBLE_DEVICES=0` 和 `--tensor-parallel-size 1`。在线客户端在本机有代理时应为环回地址设置 `NO_PROXY=127.0.0.1,localhost`，并在服务端清单中记录卡号、TP 和精度。按总吞吐和每 GPU 吞吐分别报告，TP 扩展结果与同卡数下的优化对照应分开解读。

### KV Cache INT8 参考对照

后续量化评测仅做 KV Cache 对照，历史 W8A16 结果见 [单卡文本报告](../result/qwen3-text-single-gpu/README.md)。同一模型和负载下，分别使用 `--mode baseline` 和 `--mode kv-int8-reference`。后者在页内使用 INT8 数据及每 token、每 KV head 一份 FP32 scale，按请求页表还原有效历史，再交给 FlashAttention。缓存数据、scale 和反量化工作区都占显存。这条路径用于比较容量、正确性和代价，当前使用 Eager。服务端对应选项为 `--enforce-eager --kv-cache-dtype int8-reference`。

### Qwen3-VL 图像参考测量

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.vlm \
  --model ../models/Qwen--Qwen3-VL-2B-Instruct --image example.png \
  --prompt "What is in this image?" --output ../runs/vlm.json
```

参考适配器使用 Transformers SDPA，在一张指定的 GPU 上运行。计时从已准备好的模型输入开始，包含视觉编码、生成及输出解码。报告保留图片 SHA-256、展开后的输入长度和图片网格信息。Nano-vLLM 的连续批处理、HTTP 图文入口和 VLM TP 仍待接入。

## 本地评估数据

数据保存在仓库外的 `../datasets/`，各目录的 `manifest.json` 记录来源、固定 revision、文件大小和 SHA-256。将数据加工为评测输入时，记录切分、抽样种子和样本数，结果写入 `result/`。

| 本地目录 | 来源与范围 | 用途 |
|---|---|---|
| `mmlu/` | `cais/mmlu` 的 `all` 配置，包含 dev、validation、test | 多项选择准确率。dev 可作 few-shot 示例。 |
| `sharegpt/` | `anon8231489123/ShareGPT_Vicuna_unfiltered` 的 `ShareGPT_V3_unfiltered_cleaned_split.json` | 从对话生成服务压测负载。固定筛选规则及输入和输出长度分桶，只保留必要的复现元数据。 |
| `textvqa/` | `lmms-lab-encoder/textvqa` 的完整 validation 分片 | 图像文字理解。使用参考答案计算准确率，记录图片分辨率和展开 token 数。 |

原始数据未转换为当前 SSE 压测脚本所需的 `messages` JSONL。量化质量比较应固定样本、提示词、解码参数和评分方法。
