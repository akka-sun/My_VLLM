#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp
out="$root/runs/vllm-sharegpt-pressure"
vllm="$root/.venv-vllm/bin/vllm"
model="$root/models/Qwen--Qwen3-0.6B"
dataset="$root/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
export CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_V1=1 OMP_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export TRITON_CACHE_DIR="$root/runs/vllm-kv-ab/triton-cache"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
url=http://127.0.0.1:8012
server=
stop() {
  if [[ -n "$server" ]]; then
    kill -TERM "$server" 2>/dev/null || true
    wait "$server" || true
    server=
  fi
}
trap stop EXIT
for job in auto:1536:normal fp8_e4m3:1536:normal auto:384:small fp8_e4m3:384:small fp8_e4m3:768:equal-bytes; do
  IFS=: read -r dtype pages capacity <<< "$job"
  tag="$capacity-$dtype"
  "$vllm" serve "$model" --served-model-name qwen3 --host 127.0.0.1 --port 8012 \
    --dtype bfloat16 --kv-cache-dtype "$dtype" --block-size 32 --num-gpu-blocks-override "$pages" \
    --enforce-eager --max-model-len 2048 --max-num-seqs 128 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization .25 --no-enable-prefix-caching > "$out/$tag.server.log" 2>&1 &
  server=$!
  ready=0
  for i in {1..90}; do
    if curl -fsS "$url/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$server" 2>/dev/null || break
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then echo "Server failed: $tag"; exit 1; fi
  echo "Server ready $tag"
  for concurrency in 32 64 128; do
    name="$tag-c$concurrency"
    curl -fsS "$url/metrics" | grep -E '^(vllm:num_preemptions_total|vllm:prompt_tokens_total|vllm:generation_tokens_total)' > "$out/$name.before.txt"
    "$vllm" bench serve --backend vllm --base-url "$url" --model "$model" --served-model-name qwen3 \
      --dataset-name sharegpt --dataset-path "$dataset" --num-prompts 256 --seed 42 \
      --sharegpt-output-len 128 --ignore-eos --request-rate inf --max-concurrency "$concurrency" \
      --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 --disable-tqdm \
      --ready-check-timeout-sec 30 --save-result --result-dir "$out" --result-filename "$name.json" \
      --metadata "backend=TRITON_ATTN_VLLM_V1" "kv_dtype=$dtype" "kv_blocks=$pages" "page_size=32" "seed=42" \
      > "$out/$name.bench.log" 2>&1
    curl -fsS "$url/metrics" | grep -E '^(vllm:num_preemptions_total|vllm:prompt_tokens_total|vllm:generation_tokens_total)' > "$out/$name.after.txt"
    echo "Completed $name"
  done
  stop
done
