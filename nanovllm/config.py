import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
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
        if self.quantization not in (None, "int8"):
            raise ValueError("quantization must be None or 'int8'")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.quantization == "int8" and self.hf_config.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("W8A16 requires a float16 or bfloat16 model dtype")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
