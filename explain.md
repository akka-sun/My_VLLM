# Nano-vLLM 代码讲解

## 流式输出怎么实现

本地 Python 调用和 HTTP 服务都从 `LLMEngine.step(stream=True)` 拿每轮新增的 token。本地用 `generate_stream()` 逐次返回累计文本，HTTP 服务在 `stream=true` 时逐次发送新增文本。普通的 `generate()` 会等整批请求跑完，再返回完整结果列表。

### 从输入到输出

1. **请求入队。** `LLM` 继承了 `LLMEngine`。`generate_stream()` 第一次被迭代时，通过 `add_request()` 把字符串转成 token ID，封装成 `Sequence`，放进调度器的等待队列。也可以直接传入 token ID 列表。引擎此时必须空闲。
2. **跑一轮推理。** `step(stream=True)` 调用 `Scheduler.schedule()` 选出本轮的请求，先记下每个请求已经生成了多少 token。随后 `ModelRunner.run()` 运行模型并采样，给出候选 token ID。首次处理提示词叫 prefill，后续逐步生成叫 decode。同一轮可以处理一个请求的提示词，同时继续生成另一个请求的答案。
3. **确认新 token。** `Scheduler.postprocess()` 把 token 追加到序列。长提示词若分成几段处理，前面的分段只更新缓存，最后一段处理完才追加第一个生成 token。追加后，如果达到 `max_tokens`，或者遇到未被忽略的 EOS（结束 token），调度器会标记请求完成并释放缓存块。
4. **返回本轮增量。** `step()` 用旧长度切片（`seq.completion_token_ids[previous_length:]`），只返回刚追加的 token。没有新增 token 的请求本轮没有事件。结束的那一轮照样返回最后一个 token，并带上 `finished=True`。
5. **返回累计文本。** `generate_stream()` 按 `seq_id` 累计生成的 token ID，把整个累计列表解码成文本，再用 `yield` 交给调用者。这样当一个字符的字节分散在多个 token 里时，后续结果可以修正前一轮文本的末尾。

源码入口见 [LLM](nanovllm/llm.py)、[LLMEngine](nanovllm/engine/llm_engine.py)、[Scheduler](nanovllm/engine/scheduler.py) 和 [ModelRunner](nanovllm/engine/model_runner.py)。

### 用两轮生成走一遍

假设输入是「打个招呼」，`max_tokens=2`，模型恰好先生成表示「你」的 token `101`，再生成表示「好」的 token `102`。这些 ID 只用于演示。

| 轮次 | 引擎做了什么 | 本轮 `token_ids` | 累计 `text` | `finished` |
| --- | --- | --- | --- | --- |
| Prefill | 处理完提示词，追加 `101`，从旧长度 `0` 开始切片 | `[101]` | 你 | `False` |
| Decode | 追加 `102`，从旧长度 `1` 开始切片，达到两枚生成 token 的上限 | `[102]` | 你好 | `True` |

```python
from nanovllm import SamplingParams

# llm 是已经初始化的 LLM 实例
for event in llm.generate_stream(["打个招呼"], SamplingParams(max_tokens=2)):
    print(event["text"])
```

在上述假设下，循环先打印「你」，再打印「你好」。界面逐步显示时，用最新的 `text` 替换旧文本。多个请求一起运行时，可以按 `seq_id` 分别更新各自的文本。

这条链路通过 Python 迭代器交付事件。调用者继续迭代时，引擎才会推进下一轮（同一轮的多条事件会先依次交付）。如果提前退出循环，尚未完成的请求仍留在引擎里，可以继续消费原迭代器，或在关闭它后调用 `step()` 推进剩余请求。

### HTTP 客户端怎么收到流

1. 客户端向 `/v1/chat/completions` 或 `/v1/completions` 发出带 `"stream": true` 的请求。聊天消息先通过 tokenizer 的聊天模板转换成 token ID，随后由 `AsyncLLMEngine.add_request()` 提交给引擎。
2. `AsyncLLMEngine._run()` 在后台不断调用同一个 `step(stream=True)`。它按 `seq_id` 找到请求，累计解码，然后把事件放进该请求自己的队列。
3. `respond()` 里的 `stream()` 从队列取事件，把已发送的文本前缀去掉，通过 SSE（服务端发送事件）推送剩下的文本。如果还没生成完，末尾的替换字符「�」会先缓一缓，等后续 token 补齐字符。正常结束时发送结束原因和 `data: [DONE]`。如果客户端断开连接，响应清理会取消还在运行的请求。

还是用「你」和「你好」举例。聊天接口先发送一帧标明角色是 `assistant`，接着发送 `delta.content` 为「你」的一帧。下一轮累计文本变成「你好」，已发过的「你」被去掉，因此这帧只发「好」，并带上 `finish_reason: "length"`。客户端把这两帧的内容拼起来就得到「你好」。

源码见 [AsyncLLMEngine](nanovllm/server/async_engine.py) 和 [HTTP 路由及 SSE 封装](nanovllm/server/app.py)。

## step：一轮推理在做什么

入口是 [LLMEngine.step()](nanovllm/engine/llm_engine.py)。它每次推进一批请求的一轮计算；一次调用可能处理多个 prompt token，也可能为多个请求各生成一个 token，不一定产生可返回的输出。`generate()` 和 `generate_stream()` 在外层反复调用它，直到请求结束。

### 按调用顺序看四个环节

```python
seqs, is_prefill = self.scheduler.schedule()
if not seqs:
    return [], 0
```

**调度本轮工作。** [Scheduler.schedule()](nanovllm/engine/scheduler.py) 根据 token 预算、KV 页和调度策略选择请求，并设置每条请求的 `num_scheduled_tokens`。RUNNING 同时包含部分 prefill 和 decode 请求。`is_prefill` 是整批执行路径标记：只要包含需走变长路径的请求，就使用该路径，不代表整批都在处理 prompt。例如一个请求处理 3 个 prompt token，另一个请求处理 1 个 decode token，可放进同一轮。

```python
previous_lengths = [seq.num_completion_tokens for seq in seqs] if stream else []
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

**记下旧长度，再运行模型。** [ModelRunner.run()](nanovllm/engine/model_runner.py) 准备 token ID、位置和 KV 元数据，执行模型得到 logits，再调用 Sampler 得到候选 token。例如某请求已有输出 `[101]`，旧长度就是 1；本轮候选可能是 102，但是否加入输出还要看下一步的进度判断。

```python
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

**更新进度并确认输出。** `postprocess()` 更新完整 KV 块的哈希、推进 `num_computed_tokens`，并清零本轮计划数。若 computed 还没追上当前序列长度，仅完成一个中间 chunk，候选 token 不追加到输出；追上后才追加一个新 token。达到生成上限或遇到未忽略的 EOS 时，结束请求并释放 KV 页。例如 5-token prompt 分成 3+2 两段，第一段只计算缓存，第二段才生成第一个输出 token。

```python
# stream=True 的核心切片
seq.completion_token_ids[previous_length:]
# 默认模式的筛选条件
if seq.is_finished
```

**按调用方式返回结果。** 流式模式返回新增 token ID、请求 ID 和结束标志；默认模式只返回本轮刚结束请求的完整生成 token 列表。例如累计输出从 `[101]` 变成 `[101,102]`，流式模式返回 `[102]`，而默认模式在请求结束时返回 `[101,102]`。`step()` 本身不把 token ID 解码成文本；本地生成接口或 HTTP 后台负责解码。

### 一个请求从分块到结束

假设 prompt 为 `[10,11,12,13,14]`，预算为 3，`max_tokens=2`，没有前缀命中或抢占。模型最终生成 101、102，二者均非 EOS。

| 轮次 | 本轮输入 | 计算后 computed | 累计输出 | `step(stream=True)` 的输出 |
| --- | --- | --- | --- | --- |
| 1 | prompt 的 `[10,11,12]` | 3 | `[]` | `[]`，中间 chunk 不产生输出 |
| 2 | 剩余 prompt `[13,14]` | 5 | `[101]` | 新 token `[101]`，未结束 |
| 3 | 上轮生成的 `[101]` | 6，随后释放 KV 时重置 | `[101,102]` | 新 token `[102]`，已结束 |

101 在第二轮被采样产生，但它的 KV 要等第三轮作为输入时才计算。102 产生后就达到生成上限，不必再把 102 送进模型。

返回值的第二项 `num_tokens` 用于区分吞吐统计：包含 prefill 的批次返回本轮计划 token 总数，纯 decode 批次返回请求数的负值，空批次为 0。这个例子依次为 `3、2、-1`；负数不是实际计算了负数量的 token。

## Top-k / Top-p：如何从 logits 选出 token

入口参数是 [SamplingParams](nanovllm/sampling_params.py) 的 `temperature`、`top_k` 和 `top_p`。请求创建时参数保存到 [Sequence](nanovllm/engine/sequence.py)，`ModelRunner.prepare_sample()` 按本轮请求顺序构造 GPU 参数张量，随后 `ModelRunner.run()` 将模型 logits 与这些参数一起交给 [Sampler.forward()](nanovllm/layers/sampler.py)。

logits 的形状为 `[请求数, 词表大小]`，每行对应一个请求。不同请求可有不同参数。例如同一批第一行可以 `top_k=1`，第二行可以 `top_k=0, top_p=0.9`。若整批均为默认值 `top_k=0, top_p=1`，传入的过滤器为 `None`，直接跳过排序和过滤。

### 1. 温度缩放和排序

```python
logits = logits.float() / temperatures.unsqueeze(dim=1)
sorted_logits, indices = logits.sort(dim=-1, descending=True)
```

先转 float32，再按每个请求的温度缩放。温度越低，softmax 后越集中；温度越高，越平缓。例如 logits `[2,1,0]` 除以温度 0.5 后变为 `[4,2,0]`，第一名优势增大。启用任一过滤器后才执行降序排序；`indices` 记录排序位置对应的原始 token ID。

### 2. Top-k 保留排名前 k 个

```python
ranks = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0)
remove = (top_ks.unsqueeze(1) > 0) & (ranks >= top_ks.unsqueeze(1))
sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
```

排名从 0 开始，k=2 时只保留排名 0、1。其他位置变为负无穷，后续 softmax 概率为 0。例如候选概率为 A=0.5、B=0.3、C=0.15、D=0.05，Top-k=2 后只剩 A、B，重新归一化为 0.625、0.375。

`top_k=0` 关闭 Top-k；k 大于词表大小时相当于全部保留。分数并列时仍只保留 k 个候选，但并列 token 的取舍不保证稳定。

### 3. Top-p 保留累计概率达到 p 的最短前缀

```python
sorted_probs = sorted_logits.softmax(dim=-1)
cumulative = sorted_probs.cumsum(dim=-1)
previous = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=-1)
remove = (previous >= top_ps.unsqueeze(1)) & (top_ps.unsqueeze(1) < 1)
remove[:, 0] = False
sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
```

`previous` 是“该 token 之前”的累计概率，保证跨过阈值的那一项仍保留。例如单独对 `[0.5,0.3,0.15,0.05]` 使用 p=0.7：

| token | 概率 | 包含当前项的累计概率 | 当前项之前的累计概率 | 保留？ |
| --- | --- | --- | --- | --- |
| A | 0.5 | 0.5 | 0 | 是 |
| B | 0.3 | 0.8 | 0.5 | 是，B 使累计值跨过 0.7 |
| C | 0.15 | 0.95 | 0.8 | 否 |
| D | 0.05 | 1.0 | 0.95 | 否 |

如果 p 恰好等于 0.5，只保留 A；如果 p 非常小，第一名仍保留，避免空候选。`top_p=1` 关闭 Top-p。

**两个过滤器同时启用时先 Top-k、后 Top-p。** Top-p 使用 Top-k 之后重新归一化的概率。上例同时设置 k=2、p=0.6，Top-k 后 A 的概率为 0.625，自己已超过 p，最终只剩 A；这与单独使用 p=0.6 时保留 A、B 不同。

### 4. 恢复 token ID，重新归一化，再随机采样

```python
logits = torch.empty_like(logits).scatter_(1, indices, sorted_logits)
probs = torch.softmax(logits, dim=-1)
noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
sample_tokens = probs.div_(noise).argmax(dim=-1)
```

`scatter_` 将排序后的分数放回原词表位置。例如排序第一名原本是 token 42，最终采样应返回 42，而非排名 0。最终 softmax 把剩下候选的概率重新归一化；被屏蔽项概率为 0。

采样使用独立的指数随机噪声。理想情况下，取 `argmax(probs / noise)` 等价于按概率分布随机抽取；实现中的下限用于数值保护。例如只保留 A、B，概率为 0.625、0.375，多次采样时 A 出现的比例趋近 62.5%，但单次仍可能选中 B。这里的 `argmax` 不是直接挑最大概率，而是挑加入随机性后的最大分数。只剩一个候选时，结果才确定。

### 如何配置与核对

```python
params = SamplingParams(temperature=0.6, top_k=50, top_p=0.9, max_tokens=32)
outputs = llm.generate(["解释一下 KV Cache"], params)
```

此调用逐轮产生 logits，每轮温度缩放后先保留前 50 名，再取累计概率达到 0.9 的最短前缀，最后随机选一个 token；并非一次选出 50 个输出 token。参数要求 k 为非负整数、p 在 `(0,1]`。本地 SamplingParams 要求正温度；HTTP 层对 0 或极小温度另做 top-1 转换。

相关测试见 [test_sampling.py](tests/test_sampling.py)，覆盖候选集合、Top-p 跨阈值/恰好到阈值、两个过滤器组合、默认行为和极小 p。以上概率与 token ID 是教学示例，不代表真实模型对某句 prompt 的实测预测。
