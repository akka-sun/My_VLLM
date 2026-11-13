"""Build one ragged target verification batch from per-request proposals."""
from copy import copy

import torch

from nanovllm.engine.speculative import SpeculativeResult, probabilities, draw, verify
from nanovllm.utils.context import reset_context


@torch.inference_mode()
def run_speculative(runner, seqs):
    views, proposals = [], []
    for seq in seqs:
        count = seq.num_speculative_tokens
        if count:
            ids, q, confidence = runner.drafter.propose(seq, count, len(seqs))
            if confidence is not None:
                count = runner.drafter.choose_length(confidence, len(seqs))
                ids, q = ids[:count], q[:count]
            view = copy(seq)
            view.token_ids = seq.token_ids + ids
            view.num_tokens = len(view.token_ids)
            view.last_token = view.token_ids[-1]
            view.num_scheduled_tokens = 1 + count
            proposals.append((ids, q))
        else:
            view = seq
            proposals.append(None)
        views.append(view)
    try:
        input_ids, positions = runner.prepare_prefill(views)
        hidden = runner.model(input_ids, positions)
        auxiliary = runner.model.model.aux_hidden_states
        # Compute only required rows: complete verification queries and the last
        # row of each ordinary prefill chunk. Avoid prompt-length x vocab logits.
        rows, offset = [], 0
        for view, proposal in zip(views, proposals):
            count = view.num_scheduled_tokens
            rows.extend(range(offset, offset + count) if proposal is not None else [offset + count - 1])
            offset += count
        indices = torch.tensor(rows, device=hidden.device, dtype=torch.long)
        logits = runner.model.compute_logits(hidden[indices], all_tokens=True)
        outputs, offset, row = [], 0, 0
        for seq, view, proposal in zip(seqs, views, proposals):
            count = view.num_scheduled_tokens
            if proposal is None:
                p = probabilities(logits[row], seq.temperature, seq.top_k, seq.top_p)
                output = draw(p)
                row += 1
                valid = count
            else:
                ids, q = proposal
                p = probabilities(logits[row:row + count], seq.temperature, seq.top_k, seq.top_p)
                tokens = verify(ids, q, p)
                output = SpeculativeResult(tokens)
                row += count
                valid = len(tokens)
            if auxiliary is not None:
                runner.drafter.capture(seq, seq.num_computed_tokens, auxiliary[offset:offset + valid])
            outputs.append(output)
            offset += count
        return outputs
    finally:
        runner.model.model.aux_hidden_states = None
        reset_context()
