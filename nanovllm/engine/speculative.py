"""Distribution-preserving speculative sampling, independent of CUDA kernels."""
from dataclasses import dataclass

import torch


@dataclass
class SpeculativeResult:
    # Accepted prefix followed by a correction or bonus token. The last token
    # has not been evaluated by the target and must remain pending.
    token_ids: list[int]


def probabilities(logits, temperature, top_k=0, top_p=1.0):
    if temperature == 0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(
            -1, logits.argmax(-1, keepdim=True), 1.)
    scores = logits.float() / temperature
    if top_k > 0 or top_p < 1:
        scores, indices = scores.sort(descending=True, dim=-1)
        if top_k > 0:
            scores[..., min(top_k, scores.shape[-1]):] = -torch.inf
        if top_p < 1:
            cumulative = scores.softmax(-1).cumsum(-1)
            previous = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
            remove = previous >= top_p
            remove[..., 0] = False
            scores = scores.masked_fill(remove, -torch.inf)
        scores = torch.empty_like(scores).scatter_(-1, indices, scores)
    return scores.softmax(-1)


def draw(probs):
    return int(torch.multinomial(probs, 1).item())


def verify(draft_ids, draft_probs, target_probs):
    """Use actual conditional q and p, including request sampling filters.

    On rejection sample normalized max(p-q, 0); after full acceptance sample
    one bonus from p. This also handles deterministic proposals/greedy p.
    """
    if target_probs.shape[0] != len(draft_ids) + 1 or len(draft_probs) != len(draft_ids):
        raise ValueError("verification needs N draft and N+1 target distributions")
    accepted = []
    for index, token in enumerate(draft_ids):
        p, q = target_probs[index], draft_probs[index]
        q_token = q[token]
        if q_token <= 0:
            raise ValueError("proposed token must have positive draft probability")
        ratio = (p[token] / q_token).clamp(max=1.)
        if torch.rand((), device=p.device) < ratio:
            accepted.append(token)
        else:
            residual = (p - q).clamp_min(0)
            mass = residual.sum()
            # Only a floating point roundoff corner may have zero residual.
            accepted.append(draw(residual / mass if mass > 0 else p))
            return accepted
    accepted.append(draw(target_probs[-1]))
    return accepted


def verification_length(confidences, costs):
    """Maximize expected committed tokens / profiled verification cost.

    costs[k] is the measured cycle cost for anchor + k draft queries at the
    current load; k=0 is ordinary decoding. Include the correction/bonus.
    """
    if len(costs) != len(confidences) + 1 or any(c <= 0 for c in costs):
        raise ValueError("provide positive costs for 0..N verified drafts")
    survival = confidences.float().clamp(0, 1).cumprod(0)
    expected = torch.cat((survival.new_ones(1), 1 + survival.cumsum(0)))
    return int((expected / expected.new_tensor(costs)).argmax().item())
