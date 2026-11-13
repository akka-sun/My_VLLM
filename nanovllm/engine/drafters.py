"""Stateful proposal models; target verification lives in ModelRunner."""
from collections import deque

import torch

from nanovllm.engine.speculative import probabilities, draw, verification_length


class IndependentDrafter:
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.caches = {}

    def retain(self, ids):
        for seq_id in set(self.caches) - set(ids):
            del self.caches[seq_id]

    @torch.inference_mode()
    def propose(self, seq, count, batch_size):
        cache = self.caches.get(seq.seq_id)
        if cache is not None:
            # Only committed history before the latest token may survive a
            # rejection. Crop draft KV before feeding the correction/bonus.
            cache.crop(min(cache.get_seq_length(), len(seq) - 1))
        cached = cache.get_seq_length() if cache is not None else 0
        ids = torch.tensor([seq.token_ids[cached:]], dtype=torch.long, device=self.device)
        proposed, distributions = [], []
        for _ in range(count):
            output = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            probs = probabilities(output.logits[0, -1], seq.temperature, seq.top_k, seq.top_p)
            token = draw(probs)
            proposed.append(token)
            distributions.append(probs)
            ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        self.caches[seq.seq_id] = cache
        return proposed, torch.stack(distributions), None


class DSparkDrafter:
    def __init__(self, model, config):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.options = config.speculative_config
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.slots = {}
        self.free = deque(range(config.max_num_seqs))
        # Reserve projected context storage BEFORE sizing target paged KV.
        self.context = torch.empty(
            config.max_num_seqs, config.max_model_len, model.config['hidden_size'],
            dtype=next(model.parameters()).dtype, device=self.device)
        self.lengths = {}

    def retain(self, ids):
        for seq_id in set(self.slots) - set(ids):
            self.free.append(self.slots.pop(seq_id))
            self.lengths.pop(seq_id, None)

    @torch.inference_mode()
    def capture(self, seq, start, auxiliary):
        if seq.seq_id not in self.slots:
            if not self.free:
                raise RuntimeError('DSpark context slots exhausted')
            self.slots[seq.seq_id] = self.free.popleft()
        if start == 0:
            self.lengths[seq.seq_id] = 0
        if self.lengths.get(seq.seq_id, 0) < start:
            raise RuntimeError('DSpark auxiliary context is incomplete')
        projected = self.model.project(auxiliary)
        self.context[self.slots[seq.seq_id], start:start + len(projected)].copy_(projected)
        self.lengths[seq.seq_id] = start + len(projected)

    @torch.inference_mode()
    def propose(self, seq, count, batch_size):
        c = self.model.config
        start = len(seq) - 1
        if seq.seq_id not in self.slots or self.lengths[seq.seq_id] < start:
            raise RuntimeError('DSpark proposal requires computed target context')
        anchor = c.get('sample_from_anchor', True)
        query_count = count if anchor else count + 1
        ids = torch.full((query_count,), c['mask_token_id'], device=self.device, dtype=torch.long)
        ids[0] = seq.last_token
        context = self.context[self.slots[seq.seq_id], :start]
        hidden = self.model.block(context, ids, start)
        if not anchor:
            hidden = hidden[1:]
        base = self.model.lm_head(hidden) * c.get('logit_scale', 1.)
        previous = torch.tensor(seq.last_token, device=self.device)
        pilot_previous = previous
        proposed, distributions, confidence = [], [], []
        for index in range(count):
            embed = self.model.markov_head.markov_w1(previous)
            scores = base[index] + self.model.markov_head.markov_w2(embed) * c.get('logit_scale', 1.)
            # Expand pruned vocabulary before applying target-space filters.
            logits = scores.new_full((c['vocab_size'],), -torch.inf)
            logits[self.model.d2t] = scores
            probs = probabilities(logits, seq.temperature, seq.top_k, seq.top_p)
            token = draw(probs)
            # Choose the verification budget from a deterministic pilot chain.
            # Selecting it from sampled draft history would condition q on the
            # selected length and invalidate ordinary rejection sampling.
            pilot_embed = self.model.markov_head.markov_w1(pilot_previous)
            confidence.append(self.model.confidence_head(hidden[index], pilot_embed))
            pilot_scores = base[index] + self.model.markov_head.markov_w2(pilot_embed) * c.get('logit_scale', 1.)
            pilot_previous = self.model.d2t[pilot_scores.argmax()]
            proposed.append(token)
            distributions.append(probs)
            previous = torch.tensor(token, device=self.device)
        return proposed, torch.stack(distributions), torch.stack(confidence)

    def choose_length(self, confidence, batch_size):
        costs = self.options.get('verification_costs')
        if costs is None:
            # Initial uncalibrated cost model: fixed overhead + query work.
            # The load term makes extra verification more costly at concurrency.
            costs = [1. + k * batch_size / self.max_num_batched_tokens
                     for k in range(len(confidence) + 1)]
        else:
            costs = costs[:len(confidence) + 1]
        return verification_length(confidence, costs)


def create_drafter(config, target):
    options = config.speculative_config
    if options['method'] == 'draft_model':
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        draft_config = AutoConfig.from_pretrained(options['model'])
        if draft_config.model_type != 'qwen3' or draft_config.vocab_size != config.hf_config.vocab_size:
            raise ValueError('Independent draft must be Qwen3 with the same target vocabulary')
        if draft_config.max_position_embeddings < config.max_model_len:
            raise ValueError('Draft context window is shorter than max_model_len')
        target_tokenizer = AutoTokenizer.from_pretrained(config.model)
        draft_tokenizer = AutoTokenizer.from_pretrained(options['model'])
        if (target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
                or target_tokenizer.all_special_tokens != draft_tokenizer.all_special_tokens):
            raise ValueError('Draft and target tokenizers must have identical token mappings')
        model = AutoModelForCausalLM.from_pretrained(
            options['model'], torch_dtype=config.hf_config.dtype,
            attn_implementation='sdpa', local_files_only=True).to('cuda')
        return IndependentDrafter(model)
    from nanovllm.models.dspark import DSparkModel
    model = DSparkModel.from_pretrained(options['model'], target, config.hf_config).to('cuda')
    c = model.config
    max_count = c.get('block_size', options['num_speculative_tokens'])
    if not c.get('sample_from_anchor', True):
        max_count -= 1
    if options['num_speculative_tokens'] > max_count:
        raise ValueError('num_speculative_tokens exceeds the trained DSpark block size')
    if c['vocab_size'] != config.hf_config.vocab_size:
        raise ValueError('DSpark and target vocabulary sizes must agree')
    if c.get('max_position_embeddings', config.max_model_len) < config.max_model_len:
        raise ValueError('DSpark context window is shorter than max_model_len')
    target.model.aux_layer_ids = tuple(model.target_layer_ids)
    return DSparkDrafter(model, config)
