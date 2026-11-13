import os
import math
from dataclasses import dataclass
import torch
from transformers import AutoConfig


def validate_speculative_config(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("speculative_config must be a dictionary")
    allowed = {"method", "model", "num_speculative_tokens", "verification_costs"}
    if set(value) - allowed:
        raise ValueError(f"unknown speculative options: {sorted(set(value) - allowed)}")
    if value.get("method") not in ("draft_model", "dspark"):
        raise ValueError("speculative method must be 'draft_model' or 'dspark'")
    if not isinstance(value.get("model"), str) or not value["model"]:
        raise ValueError("speculative model must be a local checkpoint directory")
    count = value.get("num_speculative_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
        raise ValueError("num_speculative_tokens must be an integer in [1, 64]")
    costs = value.get("verification_costs")
    if costs is not None:
        if (not isinstance(costs, list) or len(costs) != count + 1
                or any(isinstance(c, bool) or not isinstance(c, (int, float))
                       or not math.isfinite(c) or c <= 0 for c in costs)):
            raise ValueError("verification_costs must contain N+1 finite positive cycle costs")
    return dict(value)


def validate_scheduler_config(config):
    for name, default, minimum in (
        ("max_num_seqs", 512, 1), ("max_num_batched_tokens", 16384, 1),
        ("long_prefill_token_threshold", 0, 0),
    ):
        value = getattr(config, name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if getattr(config, "scheduling_policy", "fcfs") not in ("fcfs", "priority"):
        raise ValueError("scheduling_policy must be 'fcfs' or 'priority'")
    for name in ("long_prefill_token_threshold_adaptive", "scheduler_reserve_full_isl"):
        if not isinstance(getattr(config, name, False), bool):
            raise ValueError(f"{name} must be a boolean")
    max_len = getattr(config, "max_model_len", None)
    if max_len is not None and getattr(config, "long_prefill_token_threshold", 0) > max_len:
        raise ValueError("long_prefill_token_threshold cannot exceed max_model_len")


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    scheduling_policy: str = "fcfs"
    long_prefill_token_threshold: int = 0
    long_prefill_token_threshold_adaptive: bool = False
    scheduler_reserve_full_isl: bool = True
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    quantization: str | None = None
    speculative_config: dict | None = None
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        validate_scheduler_config(self)
        self.speculative_config = validate_speculative_config(self.speculative_config)
        if self.speculative_config:
            if self.tensor_parallel_size != 1:
                raise ValueError("speculative decoding currently requires tensor_parallel_size=1")
            if not os.path.isdir(self.speculative_config["model"]):
                raise ValueError("speculative model must be an existing local directory")
            # Verification uses variable-length multi-query Attention, not the
            # single-token decode CUDA graph. Keep this choice explicit.
            self.enforce_eager = True
        if self.quantization not in (None, "int8"):
            raise ValueError("quantization must be None or 'int8'")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if getattr(self.hf_config, 'dtype', None) is None:
            self.hf_config.dtype = getattr(self.hf_config, 'torch_dtype', None) or torch.float16
        if self.speculative_config and self.hf_config.model_type != "qwen3":
            raise ValueError("speculative decoding currently supports Qwen3 targets only")
        if self.quantization == "int8" and self.hf_config.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("W8A16 requires a float16 or bfloat16 model dtype")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        validate_scheduler_config(self)
