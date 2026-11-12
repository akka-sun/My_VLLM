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

## 2026-11-12：补充流式输出讲解和代码讲解约定

用户希望结合源码理解流式输出，并在以后提问时把通俗讲解写入仓库根目录的 `explain.md`。

- `AGENTS.md` 增加代码讲解约定。首次新建 `explain.md`，后续按主题追加或更新。例如，下次讲解调度器时可以追加新章节，保留流式输出章节。
- `explain.md` 从 `LLMEngine.generate_stream()` 的请求入队讲到模型采样、调度器更新、切片和累计解码。示例中的第一轮返回 `[101]` 和「你」，第二轮返回 `[102]` 和「你好」，并标记完成。

```python
seq.completion_token_ids[previous_length:]
```

这段现有逻辑提取本轮新增的 token ID。例如上一轮已生成 `[101]`，当前累计为 `[101, 102]`，从旧长度 1 切片后得到 `[102]`。生成器把它加入对应请求的累计列表，再解码成「你好」。

- 验证通过。检查了 Markdown 相对链接和 Git 差异，并运行 `python -m pytest -q tests/test_streaming.py`（设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 和 `USE_TF=0`），结果为 5 项通过，2 条环境依赖弃用警告。测试使用真实调度器和替代模型运行器，未执行真实模型端到端推理。
- 提交标题为 `docs: explain token streaming and record walkthrough convention`，目标分支为 `origin/main`。
