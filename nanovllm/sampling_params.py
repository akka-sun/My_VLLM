from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer (0 disables filtering)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
