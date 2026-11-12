# Nano-vLLM 代码讲解

## 流式输出怎么实现

`generate_stream()` 是一个 Python 生成器。调用者用 `for` 逐个取结果，引擎每跑一轮就把这一轮生成的内容交出来。普通的 `generate()` 会等整批请求跑完，再返回完整结果列表。

### 从输入到输出

1. **请求入队。** `LLM` 继承了 `LLMEngine`。`generate_stream()` 第一次被迭代时，通过 `add_request()` 把字符串转成 token ID，封装成 `Sequence`，放进调度器的等待队列。也可以直接传入 token ID 列表。引擎此时必须空闲。
2. **跑一轮推理。** `step(stream=True)` 调用 `Scheduler.schedule()` 选出本轮的请求，先记下每个请求已经生成了多少 token。随后 `ModelRunner.run()` 运行模型并采样，给出候选 token ID。首次处理提示词叫 prefill，后续逐步生成叫 decode。
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
