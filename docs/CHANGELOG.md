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
