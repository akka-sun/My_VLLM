import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


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
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        validate_scheduler_config(self)
        if self.quantization not in (None, "int8"):
            raise ValueError("quantization must be None or 'int8'")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.quantization == "int8" and self.hf_config.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("W8A16 requires a float16 or bfloat16 model dtype")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        validate_scheduler_config(self)
