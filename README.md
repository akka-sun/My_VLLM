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

## Top-k / Top-p sampling

Sampling filters can be configured per request:

```python
sampling_params = SamplingParams(temperature=0.6, top_k=50, top_p=0.9, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)
# A list of SamplingParams also supports different filters in the same batch.
```

`top_k=0` disables Top-k; positive integers keep at most k candidates (capped by
the vocabulary size). `top_p` must be in `(0, 1]`; `1.0` disables nucleus
filtering. Filters apply after temperature scaling, in Top-k then Top-p order.
Top-p keeps the shortest sorted prefix reaching p, including the crossing token;
at least one candidate always remains. Defaults preserve unfiltered sampling.

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

## Chunked prefill

```python
llm = LLM("/YOUR/MODEL/PATH", max_num_batched_tokens=512, max_num_seqs=32)
```

Each scheduler step runs active decode tokens first, then spends the remaining
token budget on prompt chunks. With a budget of 512 and 12 decode requests, up
to 500 prompt tokens can run in the same batch. A prompt emits its first output
token only after its final chunk. The budget also bounds pure decode batches.
Mixed batches use paged variable-length causal attention over the full cached
prefix; pure decode keeps the CUDA graph path. KV blocks for a prompt are still
reserved up front, so chunking limits compute per step, not total KV capacity.
If decode fills every token/sequence slot, waiting prompts resume when slots
become available. No throughput or latency improvement is claimed without a
measurement on the target GPU/model.

## INT8 weight-only quantization (W8A16)

```python
llm = LLM("/YOUR/MODEL/PATH", quantization="int8")
```

Load a regular FP16/BF16 safetensors checkpoint. Attention/MLP linear weights
are symmetrically quantized per output channel after tensor-parallel sharding:
INT8 values plus FP32 scales. CUDA matmul dequantizes tiles inside a Triton
kernel, keeping activations and outputs in FP16/BF16 and accumulating in FP32.
Embedding, LM head, norms, biases, and KV cache keep their floating dtype; tied
embedding/head weights stay tied. `quantization=None` preserves the default.
This is W8A16, not W8A8, GPTQ, AWQ, or a loader for pre-quantized checkpoints.
INT8 linear storage is about one byte per weight plus four bytes per output
channel; total model memory savings vary. Accuracy and speed depend on the
model and hardware, and must be measured before drawing conclusions.

## OpenAI-compatible API server

Install the optional HTTP dependencies in the same Linux/CUDA environment used
for model inference, then start one server process:

```bash
pip install ".[server]"
python -m nanovllm.server --model /YOUR/MODEL/PATH \
  --served-model-name nano-vllm --port 8000 \
  --max-num-batched-tokens 512 --max-num-seqs 32
# Optional: --quantization int8 --enforce-eager --tensor-parallel-size 2
# The installed entry point is also available: nanovllm-serve --help
```

The default host is `127.0.0.1`. One background thread owns model initialization,
tokenization, scheduling, and execution; concurrent HTTP requests enter the same
continuous batch. Use a single Uvicorn HTTP worker (the CLI sets this); tensor
parallelism is controlled by `--tensor-parallel-size`.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nano-vllm","prompt":"Hello","max_tokens":32}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nano-vllm","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true,"stream_options":{"include_usage":true}}'
```

OpenAI Python clients can point at the local server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="nano-vllm", messages=[{"role": "user", "content": "你好"}],
    max_tokens=64, stream=True,
):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Both endpoints support nonstreaming JSON and incremental SSE terminated by
`data: [DONE]`. Chat uses the model tokenizer's chat template. Responses include
`stop`/`length` finish reasons and token usage; streaming usage is opt-in via
`stream_options.include_usage`. EOS/special tokens are excluded from displayed
text but generated tokens including EOS count toward usage. Decode runs before
the next GPU step, so an already-produced token is not held behind that step.
`GET /v1/models` lists the served name; `GET /health` reports worker health.

This is a text-only compatibility subset: one string or one token-ID list for
completions, text chat messages, `n=1`, `max_tokens` (default 16), `temperature`
(0–2), `top_p`, and the `top_k` extension. Temperatures at or below `1e-10`
select greedy top-1. Batched prompts, tools, multimodal input, custom stop
strings, logprobs, penalties, and other unknown fields return an explicit 400
instead of being silently ignored. Prompt plus `max_tokens` must fit the model
context and KV capacity. Invalid fields/context return OpenAI-shaped 400 errors,
unknown models 404, overload/unavailable service 503, and execution failures 500
(or an error SSE event after headers have been sent).

Disconnects cancel queued/active requests and reclaim KV pages after any current
model step completes. Request admission and event buffers are bounded; slow
consumers are cancelled. Shutdown aborts pending requests and closes the engine.
The original offline `generate()` and `generate_stream()` remain available.

For CPU logic/HTTP tests, install the test extra in a suitable environment and
run `python -m pytest -q`. GPU kernel tests skip when CUDA/Triton/FlashAttention
are unavailable. The HTTP tests use the real engine scheduler with deterministic
model execution; they do not establish real model performance or quality.

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
