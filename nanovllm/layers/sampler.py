import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_ks: torch.Tensor | None = None, top_ps: torch.Tensor | None = None):
        logits = logits.float() / temperatures.unsqueeze(dim=1)
        if top_ks is not None or top_ps is not None:
            sorted_logits, indices = logits.sort(dim=-1, descending=True)
            if top_ks is not None:
                ranks = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0)
                remove = (top_ks.unsqueeze(1) > 0) & (ranks >= top_ks.unsqueeze(1))
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            if top_ps is not None:
                sorted_probs = sorted_logits.softmax(dim=-1)
                cumulative = sorted_probs.cumsum(dim=-1)
                # Keep the token that first reaches/crosses p, and always keep rank 0.
                previous = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=-1)
                remove = (previous >= top_ps.unsqueeze(1)) & (top_ps.unsqueeze(1) < 1)
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(1, indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
