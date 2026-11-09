<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

## 项目开发约定

本项目的优化开发仓库为 [akka-sun/My_VLLM](https://github.com/akka-sun/My_VLLM)。开发前请阅读 [项目记忆与协作约定](AGENTS.md)；每次修改的主体代码、作用解释、简单示例和验证结果统一记录在 [改动记录](docs/CHANGELOG.md) 中。完成改动后提交并推送对应版本。

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Streaming output

```python
for chunk in llm.generate_stream(["Hello, Nano-vLLM."], sampling_params):
    # text is the cumulative completion; token_ids contains only this step's new tokens.
    print(chunk["seq_id"], repr(chunk["text"]), chunk["finished"], flush=True)
```

For manual scheduling, call `add_request()` and then `step(stream=True)` until
`is_finished()`. It returns `(chunks, num_tokens)`; each chunk contains `seq_id`,
incremental `token_ids`, and `finished`. Intermediate chunked-prefill steps may
return no chunks. EOS is included in the final token delta. Default `step()` and
`generate()` keep their existing complete-result behavior.

`generate_stream()` requires an idle engine. Consume the iterator fully before
starting another generation; stopping early leaves requests queued. Its `text`
is decoded from all completion tokens so multibyte token fragments can be
resolved; replace the displayed text with this value instead of concatenating it.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
