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
