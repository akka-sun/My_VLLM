# Nano-vLLM 改动记录

本文件集中记录本项目每次修改的主体内容、解释和简单示例。协作规则见 [AGENTS.md](../AGENTS.md)。后续改动按时间顺序追加。

## 2026-11-09：建立项目记忆与统一改动记录

### 需求与主体修改

将用户的持续开发约定纳入版本控制，使后续开发者和 AI 助手能够读取并遵循。

- `AGENTS.md`：新增仓库级协作约定，明确开发目标仓库、每次改动完成后的提交推送流程，以及解释和记录要求。
- `docs/CHANGELOG.md`：新增统一改动记录，并以本条记录展示格式。
- `README.md`：增加项目协作入口，方便开发者发现上述文件。

### 关键约定、作用与示例

**提交与推送流程**

```text
实现改动 → 必要验证 → 更新统一改动记录 → Git 提交 → 推送至 akka-sun/My_VLLM
```

这段约定将代码、验证结果和解释一起保存到同一版本中。例如，用户要求优化调度器时，应完成调度逻辑修改和相关验证，写好本文件中的对应记录，再提交并推送该版本。

**解释主体代码**

```text
关键文件 / 函数 → 代表性代码段 → 作用说明 → 简单例子 → 验证结果
```

这段约定让读者理解一段逻辑解决的问题。例如，若后续修改批处理上限，应展示判断上限的核心代码，解释其如何限制批次，再用“上限为 4、队列有 6 个请求”的例子说明本轮选取哪些请求；不需要逐行复述代码。

**集中保留历史**

每次改动追加到本文件。例如，下一次优化应新增一个日期标题，保留本条记录，让读者在一个文件中追踪全部修改。

### 验证与版本信息

- 本次仅修改 Markdown 文档，不涉及推理代码或运行行为，无需运行 GPU 推理测试。
- 提交前检查 Markdown 差异、相对链接目标与必需约定，并执行 `git diff --cached --check`。
- 提交标题：`docs: establish project memory and change documentation workflow`。
- 目标仓库：`https://github.com/akka-sun/My_VLLM`；目标分支：`main`。

## 2026-11-09：增加流式 token 输出

### 需求与主体修改

- `nanovllm/engine/llm_engine.py` 的 `step(stream=True)` 在每轮返回新增 token 和结束标志；默认调用仍只返回已完成请求。
- 新增 `generate_stream()` 迭代接口，返回请求 ID、新增 token、累计文本和结束标志。README 增加使用方式。
- `tests/test_streaming.py` 使用真实调度器测试增量输出；测试环境仅替换依赖 CUDA 模型栈的 ModelRunner。

### 核心代码解释与示例

```python
previous_lengths = [seq.num_completion_tokens for seq in seqs]
# 在模型运行和 scheduler.postprocess 后：
delta = seq.completion_token_ids[previous_length:]
```

记录本轮生成前的长度，再截取新增部分。例如已有 `[7]`，本轮追加 `9`，只返回 `[9]`。分块预填充若没有追加 token，则不产生输出；结束时仍返回最后一个 token，包括 EOS。

```python
token_ids = completions.setdefault(output["seq_id"], [])
token_ids.extend(output["token_ids"])
yield {**output, "text": self.tokenizer.decode(token_ids)}
```

按请求分别累计 token，随每次 `step()` 返回立即交给调用者。例如两个请求同时生成 `A`，会各产生一个事件；下一轮各返回新增 `!`，累计文本为 `A!`。文本采用累计解码，避免独立解码半个多字节字符；调用者应替换显示文本，不能把累计文本直接拼接。

流式迭代要求引擎空闲，参数列表长度必须匹配提示数量。提前停止迭代不会取消剩余请求，需继续调用 `step()` 完成；README 和方法说明均注明此行为。

### 验证与版本信息

- 先运行新增测试，旧实现出现 4 个预期失败（缺少流式接口），1 个兼容性测试通过；实现后 `python -m pytest -q`：5 passed。
- 覆盖首 token 立即返回、EOS 仅输出一次、分块预填充、多请求增量输出、原 `generate()` 行为和参数长度校验。
- 测试产生 2 条现有环境中 botocore 的弃用警告。
- Windows 环境为 PyTorch 2.3.1+cu118、RTX 3060 Laptop，缺少 Triton / FlashAttention，且低于项目要求的 PyTorch 2.4；未执行真实模型端到端推理，无吞吐提升声明，也未下载权重。
- 提交标题：`feat: add incremental token streaming`；目标分支：`main`。

## 2026-11-09：增加按请求配置的 Top-k / Top-p 采样

### 需求与主体修改

- `nanovllm/sampling_params.py`：新增 `top_k=0`、`top_p=1.0` 及范围校验。
- `nanovllm/engine/sequence.py`：在主进程序列中保存两个参数。
- `nanovllm/engine/model_runner.py`：`prepare_sample()` 按当前调度顺序构造 GPU 参数，`run()` 将其传给 Sampler。采样只在 rank 0 执行，因此其他张量并行进程不需要序列化这些参数。
- `nanovllm/layers/sampler.py`：温度缩放后执行 Top-k、Top-p，再沿用指数噪声采样。
- `tests/test_sampling.py`：新增参数、候选集合、概率、CUDA 参数传递和计算图测试；README 增加配置示例。

### 核心代码解释与示例

**参数与请求传递**

```python
top_k: int = 0
top_p: float = 1.0
# ModelRunner.prepare_sample（示意）：
top_ks = [seq.top_k for seq in seqs]
top_ps = [seq.top_p for seq in seqs]
```

每个请求有自己的过滤参数。例如同一批请求可以分别用 `top_k=1` 和 `top_p=0.9`。`top_k` 必须为非负整数，0 关闭；`top_p` 必须在 `(0, 1]`，1 关闭。整批未启用的过滤器传 `None`，默认配置直接走原来的无排序采样路径。

**Top-k 过滤**

```python
sorted_logits, indices = logits.sort(dim=-1, descending=True)
remove = (top_ks.unsqueeze(1) > 0) & (ranks >= top_ks.unsqueeze(1))
sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
```

按概率顺序只保留前 k 个候选，过滤后的负无穷在 softmax 中变成零概率。例如原概率 `[0.6, 0.3, 0.1]`、k=2，仅前两个候选保留；归一化后约为 `[0.667, 0.333, 0]`。并列分数也只保留 k 个候选，但并列 token 的取舍不保证稳定；k 大于词表大小相当于保留全部候选。

**Top-p 最短概率前缀**

```python
sorted_probs = sorted_logits.softmax(dim=-1)
cumulative = sorted_probs.cumsum(dim=-1)
previous = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=-1)
remove = (previous >= top_ps.unsqueeze(1)) & (top_ps.unsqueeze(1) < 1)
remove[:, 0] = False
```

保留累计概率首次达到或超过 p 的最短前缀。例如概率 `[0.6, 0.3, 0.1]`、p=0.7 时保留前两个，第二个是跨过阈值的 token；p=0.6 时只需第一个。若先应用 k=2，再取 p=0.6，Top-k 归一化后第一个概率约 0.667，因此只保留第一个。极小的 p 转为 float32 后可能下溢为 0，强制保留第一名避免产生空分布。

**映射回词表并采样**

```python
logits = torch.empty_like(logits).scatter_(1, indices, sorted_logits)
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

把候选分数恢复到原 token ID 的位置，再沿用原来的随机采样方式。例如排序第一名原本是 token 42，最终返回的仍是 42，而非排序名次 0。温度缩放改为非原地运算，避免输入已为 float32 时意外修改调用者的 logits。

### 验证与版本信息

- 实现前新增采样测试出现 12 个预期失败（缺少参数或接口）；实现后全套测试最终为 `21 passed, 2 warnings`。
- 执行方式：PowerShell 中设置 `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'` 和 `$env:USE_TF='0'` 后运行 `python -m pytest -q`。
- CPU 与 RTX 3060 Laptop GPU 均验证混合批次的候选范围；验证 Top-p 跨阈值 token、恰好到达阈值、Top-k 并列/超过词表、非法参数、默认行为与输入不变性。
- 使用真实 `ModelRunner.prepare_sample()` 测试参数传到 CUDA，调用实际 `run()` 连接逻辑；仅模型执行由固定 logits 替代。
- 极小 p 的新增回归测试先复现错误，再通过第一名保留保护修复。
- 独立代码审查发现超大 `top_k` 转为 int64 时溢出的边界；增加回归覆盖并在传输前限制到 int64 最大值，仍保持超过词表大小时保留全部候选的语义。
- `torch.compile(..., backend="eager", fullgraph=True)` 在独立子进程中通过。隔离原因：本机旧 Transformers 的 `file_utils.ModelOutput` 缺失会触发 PyTorch 2.3 Dynamo 类型检查错误；未验证生产 Inductor/Triton 编译。
- `python -m compileall -q nanovllm tests` 通过。2 条警告来自环境中的 botocore 弃用 API。
- 受第一项记录中的 Windows/CUDA 依赖限制，未下载权重或运行真实模型端到端推理。启用过滤需排序，可能增加耗时；未实测性能，不声明提速。
- 提交标题：`feat: support per-request top-k and top-p sampling`；目标分支：`main`。

## 2026-11-10：Chunked Prefill 与 Decode 混合调度

### 需求、方案与主体修改

现有实现已经支持单个 prompt 分块，但等待队列总是优先于 decode，长 prompt 会阻塞已开始生成的请求。本次采用用户确认的混合批次方案：先为 decode 分配预算，再用剩余预算处理一个或多个 prefill chunk。没有采用仅在两个独立批次间轮换的方案，因为混合批次能复用已有变长 Attention 和剩余预算。

- `nanovllm/engine/scheduler.py`：重写 `schedule()` 的批次组装顺序；decode 同样受 token 和序列数上限限制，剩余空间可容纳多个部分 prompt。`postprocess()` 根据每条序列实际缓存进度决定是否追加输出。
- `nanovllm/engine/model_runner.py`：`prepare_prefill()` 同时准备 prompt chunk 和 decode 单 token，兼容张量并行 worker 中只有 `last_token` 的 decode 序列。
- `nanovllm/layers/attention.py`：明确混合批次的 paged K/V 与右下角对齐因果掩码约定；无需引入第二套 Attention 内核。
- `tests/test_chunked_prefill.py`：覆盖交错执行、两个预算、多个部分 prompt、前缀复用、抢占恢复、张量并行输入、分块 Attention 数值。
- `tests/test_attention_cuda.py`：提供使用真实 Triton/FlashAttention 的可选 GPU 数值集成测试；依赖不满足时明确跳过。
- `README.md`：补充配置方式、调度行为和 KV 容量边界。

### 核心逻辑、解释与示例

```python
seq.num_scheduled_tokens = 1  # 先调度 decode
remaining = self.max_num_batched_tokens - num_batched_tokens
seq.num_scheduled_tokens = min(num_tokens, remaining)  # 再填 prompt chunk
```

同一轮先保障正在生成的请求，再按剩余预算推进等待请求。例如预算 4，有一个 decode 和长度 8 的新 prompt，则本轮执行 1 个 decode token 与 3 个 prompt token；后续轮次继续交错，直到最后一个 chunk 才输出 prompt 的第一个生成 token。序列槽位满或 decode 用完预算时，prefill 等待空位，不保证无界请求流下的抢占公平性。

```python
if seq.num_cached_tokens < seq.num_tokens:
    continue
seq.append_token(token_id)
```

是否产生输出由每条序列决定，与整批选择的 Attention 路径分开。例如同一批中一个 decode 立即追加输出，另一个 prompt 仅完成第 3/8 个 token，后者不追加模型为中间 chunk 计算的候选 token。纯 decode 仍使用原 CUDA Graph 路径，混合批次使用变长路径。

```python
input_ids.extend(seq[start:end] if seq.token_ids else [seq.last_token])
```

主进程保存完整 token 序列，decode worker 的序列化状态仅保留最后一个 token。例如 decode 已缓存 2 个 token、最后 token 为 7，新 prompt 本轮位置为 2、3，则混合输入为 `[7, prompt[2], prompt[3]]`，位置为 `[2, 2, 3]`，每条请求的 K 长度仍包含缓存前缀。

```python
if context.block_tables is not None:
    k, v = k_cache, v_cache
```

KV 先按物理 slot 写入 cache，再通过 block table 读取完整逻辑前缀。例如逻辑页顺序 `[1, 0]`，当前查询位置 2–4，Q 长度为 3，K 长度为 5；因果掩码必须按右下角对齐，分别看到 3、4、5 个历史 token。参考：[FlashAttention 因果掩码约定](https://github.com/Dao-AILab/flash-attention#changelog)。

### 验证与版本信息

- 新增测试实现前为 4 failed / 3 passed，四个失败分别确认混合调度、decode 预算、多 prompt chunk 和 TP decode 输入缺失；实现后相关调度/流式测试为 12 passed。
- 在仓库外创建 Python 3.12 验证环境，原有测试基线为 19 passed / 2 skipped。依赖安装尚未结束时曾产生导入错误，安装完成后原测试通过，没有为该临时环境状态修改产品代码。
- 分块 Attention 使用实际 `Attention.forward()`，仅 GPU store/FlashAttention 内核以 CPU 参考函数替换；三段输出与完整 PyTorch 因果 Attention 一致，覆盖物理页重排。这不等同于真实 FlashAttention GPU 内核验证。
- 独立审查复现“部分 prompt 占用全部 KV、decode 自抢占后无法重新分配”的崩溃。新增测试先复现 `AssertionError`，再通过优先回收等待中的部分 prompt 页修复；该 prompt 稍后重新计算，已生成 token 不丢失。全套 CPU 测试最终为 27 passed / 3 skipped（包括新增可选真实 GPU Attention 测试）。`compileall` 和 `git diff --check` 通过。
- 本机验证环境使用 CPU PyTorch，无 Triton/FlashAttention，未运行真实模型推理、CUDA Graph 或多 GPU 端到端测试；不下载模型权重，不声称吞吐或延迟提升。KV 仍为整条请求预留，长 prompt 必须能放入总 KV 容量。
- 提交标题：`feat: interleave chunked prefill with decode`；目标分支：`main`。

## 2026-11-10：增加 INT8 权重量化 W8A16

### 需求、方案与主体修改

用户允许 W8A16 或 W8A8，本次采用已确认的 W8A16：线性权重 INT8，激活与输出 FP16/BF16，累加 FP32。选择权重量化以避免引入激活校准和请求级激活量化；CUDA 使用 Triton tile 内解量化，不在每次 forward 构造完整浮点权重副本。

- `nanovllm/config.py`：增加 `quantization=None | "int8"`，检查模式和量化模型的 16 位 dtype。
- `nanovllm/models/qwen3.py`、`engine/model_runner.py`：将配置传入 Attention/MLP 的 QKV、输出、gate/up 和 down 线性层，构造时直接分配 INT8 权重。
- `nanovllm/layers/linear.py`：在原分片加载路径后量化，保存输出通道对应的 scale；保留 RowParallel 的 bias/all-reduce 语义。
- `nanovllm/layers/quantization.py`：量化数学与 CPU 正确性参考；`quantization_triton.py`：CUDA 融合解量化矩阵乘。
- `nanovllm/utils/loader.py`：沿用 safetensors 和 packed weight 回调，缺少 checkpoint 时明确报错；`tests/test_quantization.py` 与 README 覆盖用法和验证。

### 核心逻辑、解释与示例

```python
maximum = weight.float().abs().amax(dim=1, keepdim=True)
scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
quantized = (weight.float() / scale).round().clamp(-127, 127).to(torch.int8)
```

每行对应一个输出通道，scale 为 FP32。例如 `[0, 1, -2]` 的 scale 约为 0.015748，INT8 为 `[0, 64, -127]`，重建约为 `[0, 1.0079, -2]`。全零行使用 scale=1，避免除零。仅支持普通浮点 checkpoint，不把预量化整数权重当作浮点权重解释。

```python
loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
self._copy_weight(param_data, loaded_weight, shard_offset)
```

先分片，再按本地行量化。QKV 和 gate/up 需要把 scale 放到对应输出片段。例如两 rank 的 gate/up 总输出 `[4, 2]`，每 rank 本地输出分别为 gate 的 2 行与 up 的 1 行，up scale 从本地偏移 2 开始；不能覆盖 gate 的 scale。RowParallel 按输入列切片，各 rank 有自己的局部 scale，完成局部矩阵乘后仍求和，bias 仅 rank 0 添加。

```python
w = (w.to(tl.float32) * scales[None, :]).to(x.dtype)
acc += tl.dot(x, w)
```

CUDA 仅重建当前矩阵 tile，在 FP32 中累加并转回激活 dtype。例如 K=80 时分三个 K=32 的 tile，最后 16 个有效元素之外使用 mask=0；不会读取越界权重。CPU 路径显式恢复浮点矩阵供测试参考，生产 CUDA 不走该分支。CUDA Graph 路径不进行 CPU 同步或加载主机权重。

```python
self.weight = nn.Parameter(torch.empty(out, inp, dtype=torch.int8), requires_grad=False)
self.register_buffer("weight_scale", torch.ones(out, 1, dtype=torch.float32))
```

例如 16×64 的线性权重，FP16 存储 2048 字节，INT8+scale 为 1024+64=1088 字节；这仅是该矩阵的存储计算，不是整模型实测。Embedding、LM head、norm、bias 与 KV Cache 保持浮点，绑定 embedding/head 不改变。

### 验证与版本信息

- 实现前新增测试为 7 failed / 1 GPU skipped；实现后量化专项为 7 passed / 1 skipped。
- 使用真实 safetensors 临时文件验证 packed 权重及 bias 加载；CPU 验证 FP16/BF16 的平均绝对误差小于测试阈值 0.07、零行有限、实际 INT8 存储、TP 本地分片和 scale 偏移、rank 1 不重复加 bias、非法模式和缺失 checkpoint。
- 提供真实 Triton CUDA 数值/CUDA Graph 测试，覆盖非整 tile 尺寸，当前 CPU 验证环境明确跳过。未进行真实模型精度、困惑度、吞吐、延迟或显存峰值测试，不宣称量化提速或整模型节省比例。
- 独立审查未发现明确的严重正确性问题，建议扩大测试。已补充非等量 Q/KV 的 GQA、不同通道行值、两个 TP 本地结果相加、真实 Qwen 构造器的量化传递与 embedding/head 绑定检查（仅替换不可用的 GPU 模块）。量化专项最终 9 passed / 6 skipped；GPU 参数包括 FP16/BF16、M=1 decode、M=33 多 tile、K=2048，以及更改输入后的 Graph replay。
- 最终全套测试为 36 passed / 9 skipped；`compileall`、`git diff --check` 通过。真实跨进程 NCCL collective 和 Qwen checkpoint 端到端推理仍未验证。
- 提交标题：`feat: add W8A16 int8 weight quantization`；目标分支：`main`。

## 2026-11-10：增加 OpenAI 兼容异步 API 与 SSE 服务

### 需求、方案与主体修改

以 FastAPI 包装 LLMEngine，提供 `/v1/completions`、`/v1/chat/completions`，支持并发异步调用和逐 token SSE。采用单个后台线程统一拥有引擎及 tokenizer，由异步命令队列接收请求；没有为每个 HTTP 请求分别调用离线 `generate_stream()`，因为后者要求引擎空闲，不能安全地共享调度状态。

- `nanovllm/server/async_engine.py`：`AsyncLLMEngine` 管理单线程 executor、有限命令/事件队列、请求准入、混合调度、增量文本、取消、错误广播与关闭。
- `nanovllm/server/app.py`：`create_app()` 和 lifespan 管理引擎；两个生成接口支持 JSON/SSE；补充 `/v1/models`、`/health`、错误处理和响应生命周期清理。
- `nanovllm/server/schemas.py`、`errors.py`：显式验证文本接口子集，使用 OpenAI 风格错误对象。
- `nanovllm/server/__main__.py`：命令行启动，支持 Chunked Prefill 预算、TP、W8A16；固定单 HTTP worker。
- `nanovllm/engine/llm_engine.py`：`add_request()` 返回 sequence ID，验证 token/context/KV 容量；增加 `abort_request()`，`exit()` 可重复调用。
- `nanovllm/engine/scheduler.py`：`abort()` 从等待/运行队列移除请求并释放 KV。
- `nanovllm/__init__.py`：延迟导入 LLM，避免查看服务帮助时提前加载 CUDA 栈。
- `pyproject.toml`：server/test 可选依赖、`nanovllm-serve` 入口、pytest 测试目录；`tests/test_api.py` 和 README 提供验证与用法。

### 核心逻辑、解释与示例

**单线程所有权与连续批处理**

```python
self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nanovllm-engine')
seq_id, count = await self.call(self._add, *payload)
outputs, _ = await self.call(self.engine.step, stream=True)
```

初始化、tokenizer、add/step/abort/exit 都在同一线程执行，HTTP 事件循环仅等待计算结果。例如 A 正在 decode、B 新提交长 prompt，B 在下一轮开始前进入同一调度器，复用第一项实现的混合批次，不创建第二个模型或并发修改 Scheduler。每轮最多消费 64 条命令后推进计算，避免准入命令持续淹没推理。

**首包实时性与文本增量**

```python
handle.token_ids.extend(output['token_ids'])
text = await self.decode(handle.token_ids)  # 必须在下一轮 GPU step 前完成
handle.queue.put_nowait({**output, 'text': text})
```

例如第一轮生成 A、第二轮 GPU 推理耗时很长，客户端仍应立即收到 A。若把 decode 留在 HTTP handler 中，它可能排在第二轮推理之后，违背流式语义；真实 TCP 测试复现该阻塞后，已按上面的顺序修复。文本使用累计解码，HTTP 只发送新增后缀；中间的 UTF-8 替换字符暂缓发送。例如第一个 token 解码 `�`、第二个补全为 `你`，客户端最终只收到 `你`。关闭 tokenizer 的空格清理；若 tokenizer 仍修改已经发送的前缀，返回明确流错误而非静默拼错文本。

**OpenAI JSON/SSE 协议与使用量**

```text
data: {"object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {"object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}
data: ... finish_reason="stop" / "length" ...
data: [DONE]
```

chat 使用 tokenizer 自带的 `apply_chat_template(..., tokenize=True, add_generation_prompt=True)`；普通 completion 使用原始文本或一条 token-ID 列表。例如输入 2 tokens、生成 A 和 EOS 共 2 tokens，usage 为 `{prompt_tokens:2, completion_tokens:2, total_tokens:4}`，显示文本不包含 EOS。`stream_options.include_usage=true` 时，在 `[DONE]` 前追加 `choices=[]` 的 usage 片段；响应 ID 与 created 在整个流中不变。参考：[OpenAI streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)、[FastAPI 生命周期](https://fastapi.tiangolo.com/advanced/events/)。

**输入验证与兼容边界**

```python
greedy = body.temperature <= 1e-10
params = SamplingParams(temperature=1.0 if greedy else body.temperature,
    top_k=1 if greedy else body.top_k, ...)
```

温度 0 或极小正数转换为 top-1，不触发原 SamplingParams 的正温度断言。普通文本/整数 token 输入、文本 chat、n=1、max_tokens、temperature、top_p、top_k 和 stream 是支持的子集；批量 prompt、多模态、tools、非空 stop、logprobs、penalties 等未实现字段返回 400。未知模型返回 404；超出 context/KV 或非法 token 返回 400；超载/失效服务返回 503。完整接口规范中的其他能力未声称支持。

```python
if len(prompt) + max_tokens > config.max_model_len:
    raise ValueError(...)
if len(prompt) + max_tokens - 1 > total_kv_capacity:
    raise ValueError(...)
```

例如上下文上限 32，prompt 长 32、max_tokens=1 被拒绝；单条请求也不能大于总 KV 容量，以免调度器无限抢占或在 GPU 中越界。最后一个输出 token 不再送入模型，因此 KV 上限计算减 1。请求 ID 使 API 能精确取消某条请求，原离线返回内容保持兼容。

**断连、背压、异常与关闭**

```python
class CancellableStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.cleanup()
```

例如客户端在长 prompt 的第一个输出之前断开，甚至发送响应头就抛 OSError，也会取消该请求并回收页；仅在流生成器写 finally 不够，因为生成器可能尚未启动。取消等当前正在执行的 GPU step 返回后生效，不能中断已提交内核。已经部分输出后断连不返回“成功完成”的 usage。每请求事件缓冲默认 128、活动请求默认 256，慢消费者被取消；模型异常会给活动请求结构化错误并将服务标为失效，后续请求返回 503，细节只记入服务日志。关闭时终止剩余请求并释放引擎，重复 exit/close 安全。

### 验证与版本信息

- API 第一轮新增测试为 4 failed；实现基础路径后为 4 passed。后续真实 HTTP 和响应头失败测试分别复现首 token 被下一轮推理阻塞、尚未进入流生成器时的取消泄漏；修复后全部通过。审查另发现 `temperature=1e-12` 触发断言，回归测试先失败，再通过 near-zero greedy 转换修复。
- API 专项最终 10 passed：真实 Scheduler/BlockManager + 确定性模型执行，检查 engine/tokenizer 线程所有权、并发混合批次、queued/partial cancellation、KV 回收、早期取消准入、慢消费者、关闭、EOS/length、usage、UTF-8 分片、普通/流式错误、未知模型/非法字段/context。
- 使用真实 Uvicorn TCP 服务验证首个 SSE A 在阻塞第二轮 decode 时到达，阻塞期间 `/health` 仍响应；随后断开客户端，确认未生成到 max_tokens 就取消，线程/引擎关闭。服务只存在于测试期间，没有留下常驻服务器。
- 使用已安装的 OpenAI Python SDK 通过本地 ASGI transport 验证 completion JSON 和 chat SSE/usage 可解析；`python -m nanovllm.server --help` 在缺少 CUDA/Triton 的环境中通过。
- 测试中的模型执行为可控替代；没有模型权重或真实 GPU HTTP 端到端推理，性能/精度仍未测量。真实 GPU 内核测试保持明确 skip。
- 最终 `python -m pytest -q`：46 passed / 9 skipped；`python -m compileall -q nanovllm tests`、`git diff --check` 通过。验证环境为 Python 3.12、CPU PyTorch 2.15、Transformers 5.22、FastAPI 0.162，OpenAI SDK 3.48；未覆盖所有依赖最低版本组合。
- 提交标题：`feat: serve OpenAI-compatible async completions API`；目标分支：`main`。
