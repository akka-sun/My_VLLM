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

## Chunked Prefill：长 prompt 如何分轮计算

核心是：**每轮只计算 prompt 尚未完成的一段，并保存这段的 KV；下一轮计算新的一段时，Attention 仍能读取全部已计算前缀。** chunk 不是相互独立的小 prompt，也不会每算完一块就生成一段回答。

调用链为 `LLMEngine.step()` → `Scheduler.schedule()` → `BlockManager.allocate_slots()` → `ModelRunner.run()/prepare_prefill()` → `Attention.forward()` → `Scheduler.postprocess()`。入口和相关实现分别见 [引擎](nanovllm/engine/llm_engine.py)、[调度器](nanovllm/engine/scheduler.py)、[KV 页管理](nanovllm/engine/block_manager.py)、[模型执行器](nanovllm/engine/model_runner.py) 和 [Attention](nanovllm/layers/attention.py)。

### 1. 用进度游标记录算到哪里

```python
pending = seq.num_tokens - seq.num_computed_tokens
count = min(pending, budget, threshold or pending)
seq.num_scheduled_tokens = count
```

`num_tokens` 是当前历史总长度，`num_computed_tokens` 是已有 KV 的前缀长度，`num_scheduled_tokens` 是本轮计划计算的长度。`Sequence.num_cached_tokens` 是 computed 的兼容别名，模型执行器仍使用这个名字。例如 prompt 长 8，computed=2，剩余预算 3，生效的阈值为 2，则本轮计算 2 tokens：零基位置 2、3，不重新计算位置 0、1。

调度器在首次准入时就把请求移入 RUNNING，后续 chunk 继续推进同一个请求。RUNNING 同时包含部分 prefill 和 decode；先按现有入场顺序处理 RUNNING，再用剩余预算准入 WAITING。因此当前代码没有每轮重新强制把所有 decode 排到最前。

### 2. chunk 长度由本轮预算决定

`max_num_batched_tokens` 是整批 token 上限，`long_prefill_token_threshold` 是多个请求同时存在时每条请求的额外上限。阈值 0 关闭该限制；只剩一个请求时忽略阈值，允许用满预算。开启 adaptive 时，有效阈值至少为 `batch_budget // eligible_requests`。

例如预算 4、阈值 2，B 已在 decode，A 是新来的 8-token prompt，且 B 在以下四轮中始终未结束：

| 轮次 | B 本轮工作 | A 本轮 prompt 位置 | A 计算后进度 | A 是否产生输出 |
| --- | --- | --- | --- | --- |
| 1 | 1 个 decode token | 0、1 | 2/8 | 否 |
| 2 | 1 个 decode token | 2、3 | 4/8 | 否 |
| 3 | 1 个 decode token | 4、5 | 6/8 | 否 |
| 4 | 1 个 decode token | 6、7 | 8/8 | 是，采样首个新 token |

每轮合计 3 tokens，小于预算 4，是因为 A 的阈值限制为 2；预算是上限，不要求必须填满。如果 B 提前结束，后续仅剩 A，A 便可使用整轮预算。A 产生首个输出后，通常变成 pending=1，与其他 decode 一样每轮继续推进一个 token。

### 3. KV 页只分配到本轮计算末尾

```python
end = computed + num_new_tokens
num_new_blocks = max(0, ceil(end / block_size) - existing_blocks)
```

上面是 `allocate_slots()` 的分配公式示意，实际代码用整数除法向上取整。例如 prompt 长 1024、块大小 256、每轮处理 128：第一轮分配 1 页，第二轮复用这一页，第三轮才增长到 2 页。chunk 的长度和物理页大小是两回事；同一页可以由多个 chunk 逐步填满。

默认 `scheduler_reserve_full_isl=True` 会在准入时检查完整输入的容量，但不会提前占用所有未来页。已有完整前缀页可被命中并按引用计数共享；先固定命中页，再分配新页，避免新分配淘汰刚命中的缓存。例如两个请求共享前 256 tokens，只需共同持有一份该前缀页，后续未完成页各自拥有。

GPU KV 池本身已由模型执行器预分配，增量管理改善的是池内空闲页可用性，不代表进程分配的 GPU 显存自动减少。

### 4. 模型只接收当前 chunk，位置编号保持连续

```python
start = seq.num_cached_tokens
end = start + seq.num_scheduled_tokens
input_ids.extend(seq[start:end] if seq.token_ids else [seq.last_token])
positions.extend(range(start, end))
seqlen_q = seq.num_scheduled_tokens
seqlen_k = end
```

`prepare_prefill()` 只抽取这次需要计算的 token，位置不从 0 重新开始。例如 A 的 prompt 为 `[10,11,12,13,14,15,16,17]`，第二个 chunk 输入是 `[12,13]`、位置为 `[2,3]`，Q 长度为 2，但 K/V 的有效总长度为 4。`cu_seqlens_q/k` 记录每条请求的边界，使一个扁平批次仍能区分不同请求；`slot_mapping` 标明新 KV 应写入哪些物理位置。

混合批次中的 decode 请求也可用这条变长路径：其 Q 长度为 1，K 长度为整个上下文。TP worker 对普通 decode 仅保存最后一个 token，因此输入准备保留 `[seq.last_token]` 分支；需要重放历史的单 token chunk 则保留完整历史。

### 5. Attention 用当前 Q 读取完整缓存前缀

```python
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
if context.block_tables is not None:
    k, v = k_cache, v_cache
# 随后调用 flash_attn_varlen_func(..., causal=True, block_table=...)
```

这是 `Attention.forward()` 的关键流程摘录：有实际 KV cache 时，先写入当前 chunk 的 K/V；有历史前缀时，通过 block table 从缓存读取完整 K/V。第二个 chunk 的 Q 虽然只有位置 2、3，Attention 却能看到位置 0–3 的 K/V，因果掩码分别允许位置 2 看 0–2、位置 3 看 0–3。这样后一个 chunk 能接上前面的上下文，而不是只在 `[12,13]` 内做 Attention。

当前实现混合批次使用变长 Attention，纯 decode 沿用原 decode/CUDA Graph 路径。分块减少每轮新增查询量，但当前 chunk 仍要读取其可见历史；不等于每个 chunk 的计算成本都恒定。

### 6. 最后一个 chunk 才确认生成 token

```python
self.block_manager.hash_blocks(seq)
seq.num_computed_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
if seq.num_computed_tokens < seq.num_tokens:
    continue
seq.append_token(token_id)
```

`postprocess()` 发布已经计算完的完整 KV 块，再推进游标。模型执行器当前会给中间 chunk 也计算候选 token，但只要尚未算完当前历史，该候选就被忽略。例如 A 的前 3 个 chunk 不追加输出，第四轮 computed 追上 8，才把候选 101 追加到序列。此时长度变成 9、computed 仍为 8，下一轮输入 101，计算它的 KV，再采样下一个 token。

若 KV 不足，调度器按 FCFS 或 priority 抢占请求，释放页引用、重置 computed 并保留已有输出，稍后重算；重放中间 chunk 同样不重复输出。例如已经生成 `[101]` 后被抢占，重算历史不会重新向流式客户端发送 101，追上历史后才产生新的输出。

### 用法与验证入口

```python
llm = LLM(
    model_path, max_num_batched_tokens=512,
    long_prefill_token_threshold=128,
)
outputs = llm.generate([long_prompt, short_prompt], SamplingParams(max_tokens=32))
```

512 限制整批本轮工作量；多个请求同时存在时，128 限制单请求占用的 token 数量，不是把 prompt 预先切成固定大小字符串。分块以 token 为单位，实际长度还取决于缓存命中、剩余预算和剩余历史。

相关测试见 [test_scheduler_v1.py](tests/test_scheduler_v1.py)、[test_chunked_prefill.py](tests/test_chunked_prefill.py) 和 [test_attention_cuda.py](tests/test_attention_cuda.py)。它们分别覆盖状态/分配/抢占、混合输入与分块 Attention 参考数值、可选真实 GPU Attention。CPU 参考测试不等同于真实 GPU 性能验证。

## 混合批处理：prefill 和 decode 如何放进同一轮

**混合批处理就是同一次模型执行中，既包含某些请求的 prefill chunk，也包含另一些请求的 decode token。当前项目已实现。** 多个请求全在 decode 只是普通的多请求批处理；这里的“混合”特指同批包含不同计算阶段。

例如本轮预算 4，选择 A 的 3 个 prompt token 和 B 的 1 个 decode token，合成 4-token 输入后运行一次模型，不必等 A 的整个 prompt 完成才继续 B。A、B 是不同请求；同一请求仍须先计算完自己的 prompt，才能开始生成答案。

### 1. 调度器将两类请求加入同一个列表

源码：[Scheduler.schedule()](nanovllm/engine/scheduler.py)。RUNNING 包含部分 prefill 和 decode，两者都按 pending token 数计算本轮工作量，准入的新请求也加入同一个 `scheduled_seqs`。

```python
seq.num_scheduled_tokens = count
scheduled_seqs.append(seq)
# 返回整批及执行路径标记
return scheduled_seqs, any(seq.is_prefill for seq in scheduled_seqs)
```

例如本轮 `seqs=[A,B]`，对应计划数为 `[3,1]`。只要 A 需要变长路径，整批标记为 True；这不改变 B 的请求状态，也不是把 B 的完整历史重新计算一次。RUNNING 保持入场顺序，混合能否形成仍取决于 token 预算、活动槽位、KV 容量和阈值；不能保证每轮都混合。

### 2. 拼接输入，同时保留各自的请求边界

源码：[ModelRunner.prepare_prefill()](nanovllm/engine/model_runner.py)。假设 A 已计算前 2 个 token、prompt 总长 8，B 的前 6 个 token 已有 KV：

| 请求 | 本轮输入 token ID | 位置 | Q 长度 | 有效 K/V 总长度 |
| --- | --- | --- | --- | --- |
| A：prefill | `[12,13,14]` | `[2,3,4]` | 3 | 5 |
| B：decode | `[27]` | `[6]` | 1 | 7 |

这些 ID 仅用于教学。模型执行器合成：

```text
input_ids    = [12,13,14,27]
positions    = [ 2, 3, 4, 6]
cu_seqlens_q = [0,3,4]
cu_seqlens_k = [0,5,12]
```

`cu_seqlens_q` 表示前三个查询属于 A，第四个查询属于 B；`cu_seqlens_k` 中 5、12 是累计边界，B 的实际 K 长度为 12−5=7，不是 12。两条请求还有各自的 block table，slot mapping 分别指定新 KV 的写入位置。例如页大小 256，A 使用物理页 2、B 使用页 5，新 KV 分别写入 `[514,515,516]` 和 `[1286]`。

### 3. 共用模型计算，但 Attention 上下文独立

源码：[ModelRunner.run()](nanovllm/engine/model_runner.py) 和 [Attention.forward()](nanovllm/layers/attention.py)。`LLMEngine.step()` 对整批调用一次 `run`；线性层、MLP 等对拼接后的 token 张量执行批量运算。混合批次使用变长 Attention，按请求边界与 block table 读取 KV。

在上例中，A 的位置 4 只看 A 的位置 0–4，B 的位置 6 只看 B 的位置 0–6。虽然输入扁平拼接，B 不会因此看见 A 的 prompt。批处理共享模型权重，不表示请求之间混合上下文；相同前缀的显式缓存复用是另一项独立机制。

同批执行也不意味着两条独立 CUDA stream 必然同时运行。当前项目是一个混合 forward，并没有为 prefill/decode 建立独立 GPU 流水线或分离部署。较大的 prefill chunk 仍可能延长本轮时间，从而影响 decode 的 token 间隔；分块预算用于控制每轮工作量，不提供性能隔离。

### 4. 每个请求分别取候选、更新和返回

源码：[ParallelLMHead.forward()](nanovllm/layers/embed_head.py) 和 `Scheduler.postprocess()`。

```python
last_indices = context.cu_seqlens_q[1:] - 1
x = x[last_indices].contiguous()
```

上例的下标为 `[2,3]`，分别取 A 的最后一个查询和 B 的查询，得到两条请求的 logits，再各自采样。A 虽然输入 3 tokens，但这一步不会产生 3 个回答 token。

随后 A 的 computed 从 2 变为 5，尚未追上长度 8，其候选忽略；B 的 computed 从 6 变为 7，已追上当前历史，可追加候选 28。于是本轮 `step(stream=True)` 只返回 B 的新增 `[28]`。等 A 完成最后一个 prompt chunk，才返回 A 的第一个新 token。

实现与验证入口分别是 [混合调度测试](tests/test_chunked_prefill.py)、[统一进度与运行顺序测试](tests/test_scheduler_v1.py)。其中 `test_mixed_tp_batch_builds_positions_slots_and_full_context` 调用实际输入准备代码，验证混合输入、位置、slot、Q/K 边界和 TP decode 序列；本机测试替换 GPU 传输，不代表真实 GPU 并发或性能实测。

## 从加载模型到返回文本：两个请求走完全流程

这一节把前面的概念连起来：`LLM(...)` 先完成一次性初始化，`generate()` 将请求入队并反复调用 `step()`；每一步调度一批 token，送进同一个 Qwen3 模型，经过 Embedding、多个 Decoder Layer、LM Head 和采样，再更新请求，最后将生成的 token 解码成文本。

### 0. 先固定例子的输入和参数

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    model_path, tensor_parallel_size=1, enforce_eager=True,
    max_num_batched_tokens=3, max_num_seqs=4,
)
params = SamplingParams(temperature=1.0, top_k=2, top_p=0.9, max_tokens=2)
outputs = llm.generate(["打个招呼", "Say hi"], params, use_tqdm=False)
```

用很小的预算是为了看清分块，实际部署应根据模型和 GPU 测量后配置。此例使用单 GPU、普通浮点权重、无前缀命中/抢占，关闭 CUDA Graph 重放；`enforce_eager=True` 不意味着取消各层原有的 `torch.compile` 装饰。

为便于跟踪，人为约定以下 tokenizer 和采样结果；它们不是某个真实模型对这些字符串的预测：

| 请求 | 编码后的 prompt | 依次生成的 token | 解码结果 |
| --- | --- | --- | --- |
| A | `[11,12,13,14]` | `[101,102]` | 你好 |
| B | `[21,22]` | `[201,202]` | Hi! |

假设 EOS=2，上述生成 token 都不是 EOS，两个请求均由 `max_tokens=2` 结束。实际 tokenizer 的切分及 token ID 不一定与汉字、单词一一对应。

另假设教学小模型有 2 层，hidden size=16、query heads=2、KV heads=1、head dimension=8、MLP intermediate size=32、词表大小=512，并使用无 QKV bias 的分支。下面用这些尺寸展示数据形状；真实尺寸取自模型目录的配置，不是由上述生成参数设定。

### 1. 初始化：模型、权重、KV 池和调度器

入口 [LLM](nanovllm/llm.py) 直接继承 [LLMEngine](nanovllm/engine/llm_engine.py)，初始化按以下顺序执行：

1. `Config` 读取本地模型配置，验证调度选项，限制最大上下文，并设置 KV 块大小（默认 256）。例如确认一个请求的 prompt 加输出上限不能超过配置允许的长度。
2. 创建 `ModelRunner`，设置 NCCL 进程组、GPU 和模型浮点 dtype；即便本例 TP=1，当前代码也初始化进程组。TP>1 时另创建 worker 进程。
3. 构造 [Qwen3ForCausalLM](nanovllm/models/qwen3.py)，再由 [load_model()](nanovllm/utils/loader.py) 从 safetensors 读取权重。Q/K/V 合并加载到 `qkv_proj`，gate/up 合并到 `gate_up_proj`；权重是训练好的参数，不是在每轮生成时重新训练。例如教学模型的 embedding 矩阵形状为 `[512,16]`。
4. 执行 warmup，测量可用于 KV 的内存，再创建 KV 池，并把每层的 cache 视图交给对应 Attention。池形状为 `[2, layers, blocks, block_size, kv_heads, head_dim]`，开头的 2 表示 K、V。本例每个 token 在每层各保存一个 8 维 K 和一个 8 维 V。
5. 如未 enforce eager，则准备 decode CUDA Graph；本例跳过。随后加载 tokenizer，记录 EOS，创建 Scheduler/BlockManager；BlockManager 管理池的页编号和引用，而不是另建一套 GPU KV 数据。

模型和 KV 池准备好后，可以重复处理多个请求，不必每次 `generate()` 都重载权重。

### 2. generate：编码输入，构造 Sequence，进入 WAITING

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

`add_request()` 将字符串交给 tokenizer.encode，检查 token、上下文和总 KV 容量，再构造 Sequence。本地普通 `generate()` 不自动套用聊天模板；HTTP chat 接口会另外调用 `apply_chat_template()`。

入队后的主要状态为：

| 字段 | A | B |
| --- | --- | --- |
| token_ids | `[11,12,13,14]` | `[21,22]` |
| num_prompt_tokens / num_tokens | 4 / 4 | 2 / 2 |
| num_computed_tokens | 0 | 0 |
| num_completion_tokens | 0 | 0 |
| block_table | `[]` | `[]` |
| status | WAITING | WAITING |

这些状态由 [Sequence](nanovllm/engine/sequence.py) 保存。例如 A 虽然已经知道 4 个 prompt token 的 ID，却还没有它们的 KV，computed 因此为 0，而不是 4。

### 3. 第一轮 step：只计算 A 的前 3 个 tokens

默认 FCFS，整轮预算 3。A 无缓存命中，计划 `[11,12,13]`，预算耗尽，B 暂留 WAITING。A 准入后进入 RUNNING，分配一个 256-token 物理页，假设页号为 0。完整输入准入检查与本轮增量分配分开；这里 A 的 4-token prompt 本身也只需要一页。

`prepare_prefill()` 生成：

```text
input_ids    = [11,12,13]
positions    = [0,1,2]
cu_seqlens_q = [0,3]
cu_seqlens_k = [0,3]
slot_mapping = [0,1,2]
```

第一轮没有历史前缀，Q/K 有效长度都为 3。三条输入经过下面第 4 节中的完整模型计算，每层写入自己的 KV。

模型执行器虽产生一个候选 token，但 `postprocess()` 更新后 A 的 computed=3，小于当前长度 4，候选不追加。此时 A 输出仍为空，B 未计算。`step(stream=True)` 返回空事件；默认 `step()` 也没有已结束请求可返回，统计值为 3。

### 4. 一次模型 forward 内部到底算什么

源码主要位于 [qwen3.py](nanovllm/models/qwen3.py)。用 `T` 表示本轮所有请求拼接后的输入 token 数，前两轮本例都为 T=3。各 token 共享训练好的矩阵，但请求边界限制 Attention 的上下文。

**Embedding：token ID 查表变成向量。**

```python
hidden_states = self.embed_tokens(input_ids)
```

矩阵 `E` 为 `[512,16]`，输入 `[11,12,13]` 得到 `E[11]、E[12]、E[13]`，张量从 `[3]` 变为 `[3,16]`。这些是数值向量，还不是文本或预测概率。

**进入每个 Decoder Layer：归一化、Attention、残差、MLP。**

逻辑上可写成以下公式，源码将部分残差相加与下一次 RMSNorm 融合执行，并通过 `residual` 变量传递：

```text
y = x + Attention(RMSNorm(x))
z = y + MLP(RMSNorm(y))
```

例如第一层输入 `[3,16]`，Attention 和 MLP 后仍为 `[3,16]`，供第二层继续计算；各层权重和 KV 不同，不是在同一份 KV 上重复做两遍相同运算。

**Attention 内部：QKV 投影、Q/K Norm、RoPE、缓存和加权求和。**

```text
输入 [3,16]
  → qkv_proj 得到 [3,32]
  → Q:[3,2,8]，K:[3,1,8]，V:[3,1,8]
  → Q/K 分别做 head 维 RMSNorm（本例 qkv_bias=False）
  → 根据 positions 对 Q、K 应用 RoPE
  → 把新 K/V 写入本层缓存
  → 对各请求执行因果 Attention
  → 输出 [3,2,8]，展平并经 o_proj 回到 [3,16]
```

投影维度 32 来自 `2×8 + 1×8 + 1×8`。这是 GQA：两个 query heads 共享一个 KV head，缓存不需要存两份相同的 K/V。RoPE 对 Q/K 编入位置，V 不做这个旋转。Attention 的数学含义为：

```text
softmax(Q Kᵀ / sqrt(8) + causal_mask) V
```

例如第一轮位置 2 的查询可读取位置 0、1、2，不能看到尚未计算的位置 3。后续 chunk 通过 KV 页表读取更长的前缀，历史 token 不必再次经过 QKV 投影和 MLP，但当前查询仍需与可见 KV 做 Attention。实际 FlashAttention 内核融合计算，不必显式存下整个 QKᵀ 矩阵。

**MLP：升维、门控激活、降维。**

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)  # silu(gate) * up
x = self.down_proj(x)
```

本例 `[3,16] → [3,64]`，拆成 gate/up 两份 `[3,32]`，SiLU 门控乘法后仍为 `[3,32]`，down projection 回到 `[3,16]`。它与 Attention 一起逐层变换 token 表示。

**最后 RMSNorm 和 LM Head：向量变成词表分数。**

所有 Decoder Layer 结束后，执行最后的残差相加/RMSNorm。prefill/混合路径再选每个请求最后一个查询位置：

```python
last_indices = context.cu_seqlens_q[1:] - 1
x = x[last_indices].contiguous()
logits = F.linear(x, self.weight)
```

第一轮只有 A，选输入下标 2，`[3,16] → [1,16] → [1,512]`。512 个 logits 是对下一 token 的分数，不是 512 个生成结果。未完成 prompt 时，当前执行路径也会计算和采样这个候选，随后由 postprocess 忽略。

### 5. 第二轮 step：A 的最后一段和 B 的 prompt 同批计算

A 在 RUNNING，先安排其剩下的 1 token，随后用剩余预算 2 准入 B。B 分配物理页 1。输入为：

```text
input_ids    = [14,21,22]
positions    = [ 3, 0, 1]
cu_seqlens_q = [0,1,3]
cu_seqlens_k = [0,4,6]
slot_mapping = [3,256,257]
```

A 的 Q 长度为 1、K 长度为 4，B 的 Q/K 长度为 2。B 的 K 长度是累计边界的 `6−4=2`，不是 6。block table 为各自的一行页号 `[0]`、`[1]`。每层先写当前 KV，再让 A 的位置 3 看到 A 的 0–3，B 的位置 1 看到 B 的 0–1；两请求不会跨边界互相注意。

模型仍计算 T=3 个输入。LM Head 的 `last_indices=[0,2]`，选择 A 的 14 和 B 的 22 对应的最终表示，得到 `[2,512]` logits，分别用于预测 A、B 的下一 token。

### 6. Sampler：从两行 logits 各取一个 token

顺序为温度缩放、Top-k、Top-p、最终 softmax 和随机采样。本例假设 A 本轮最高三个候选概率为 `[0.6,0.25,0.15]`：k=2 去掉第三项，归一化为约 `[0.706,0.294]`；p=0.9 保留前两项，因为第一项还不够 0.9。随机采样假设抽到 101。B 的那行独立过滤和采样，假设得到 201。实际概率由当前模型 logits 决定，这些数值是示意。

Sampler 返回 `[101,201]` 后，postprocess 按同一请求顺序更新：

| 请求 | 本轮计算后 computed | 追加后的 num_tokens | completion_token_ids | 是否结束 |
| --- | --- | --- | --- | --- |
| A | 4 | 5 | `[101]` | 否 |
| B | 2 | 3 | `[201]` | 否 |

computed 不会因为刚采样出新 token 就立即增加；101、201 还没有作为输入计算过。默认 `step()` 此轮仍不返回完整结果，因为两者都未结束；`step(stream=True)` 则立即返回 `[101]` 和 `[201]` 两条事件。

### 7. 第三轮 step：两个请求一起 decode

现在 A 的 pending 为 `5−4=1`，B 为 `3−2=1`。本轮两个请求都是 decode，走 `prepare_decode()`：

```text
input_ids    = [101,201]
positions    = [  4,  2]
context_lens = [  5,  3]
slot_mapping = [4,258]
block_tables = [[0],[1]]
```

这轮 T=2，Embedding 为 `[2,16]`，每层只为这两个新输入计算 Q/K/V、写入新 KV，再分别读取已有上下文。A 的位置 4 能看到 A 的 0–4；B 的位置 2 能看到 B 的 0–2。不会重新把 `[11,12,13,14]` 或 `[21,22]` 全部送进模型。

纯 decode 不用 prefill 的末位置切片，每条输入本身就是一条请求的最后位置，直接得到 `[2,512]` logits。假设分别采样出 102、202。两请求达到生成上限 2，标记 FINISHED，从 RUNNING 移除并释放页引用。

计算完成时 A 的 computed 为 5，B 为 3，随后 deallocate 将游标重置为 0。A 的完整历史是 `[11,12,13,14,101,102]`，但最后的 102 不再需要计算 KV，因为请求已结束。B 的 202 同理。页回到空闲池，不是删除 GPU KV 池；本例长度未满 256，没有形成可发布的完整前缀缓存块。

### 8. generate 收集完成结果并解码

默认 `step()` 第三轮返回两条完整生成 token 列表，`generate()` 以 seq_id 收集并排序，调用 tokenizer.decode 得到：

```python
[
    {"text": "你好", "token_ids": [101, 102]},
    {"text": "Hi!", "token_ids": [201, 202]},
]
```

输出只有生成部分，不包含 prompt。若是遇到 EOS 提前结束，则也会释放缓存；默认本地 decode 如何显示特殊 token 取决于 tokenizer，不能笼统认为所有本地返回文本都自动去掉 EOS。

整个三轮执行可归纳为：

| step | 模型输入 | 本轮追加输出 | 默认 step 返回 | stream=True 返回增量 | num_tokens |
| --- | --- | --- | --- | --- | --- |
| 1 | A:`[11,12,13]` | 无 | `[]` | `[]` | 3 |
| 2 | A:`[14]`、B:`[21,22]` | A:101、B:201 | `[]` | `[101]`、`[201]` | 3 |
| 3 | A:`[101]`、B:`[201]` | A:102、B:202 | 两条完整结果 | `[102]`、`[202]`，均结束 | -2 |

负值 -2 是纯 decode 的统计标记，不是处理了负数 token。引擎请求都结束后退出 generate 循环；模型实例可继续复用。最终关闭引擎时才调用 ModelRunner.exit，结束通信/worker 等资源。

### 9. 其他配置会改变哪一段

| 配置/场景 | 对上述流程的影响 |
| --- | --- |
| 前缀缓存命中 | 从命中的完整 KV 前缀末尾开始安排新查询，少算已经有 KV 的历史。 |
| 同时存在更长请求 | 按总预算、阈值和队列状态分块；实际每轮输入不一定固定长度。 |
| KV 不足/请求取消 | 抢占保留输出并重算未缓存历史；取消直接移除请求、释放引用。 |
| CUDA Graph | eligible 的纯 decode 在 run_model 中重放捕获图；prefill/混合批次走现有变长路径。 |
| TP>1 | 各 rank 运行权重分片，rank 0 广播执行信息；行并行结果 all-reduce、词表 logits gather，主 rank 采样。 |
| W8A16 | Attention/MLP 的局部权重分片 INT8 存储、tile 内解量化；激活仍为 16 位，调度与 token 生命周期不变。 |
| HTTP 流式接口 | 异步后台持续调用 step，并先累计解码再投递事件；路由将新增文本封装为 SSE。 |

调度/状态例子已用真实 LLMEngine、Scheduler、BlockManager 和固定 tokenizer/模型输出替代执行器逐轮核对，确认输入、进度、页号、输出和释放行为。Decoder Layer 的说明对照实际源码给出数学含义与形状，不代表运行了该教学模型的 GPU 前向，也不是对真实模型输出或性能的测量。
