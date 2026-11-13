"""Dense Qwen3 DSpark reference implementation using PyTorch SDPA.

Implements the public Qwen3DSparkModel checkpoint contract (parallel DFlash
backbone, vanilla low-rank Markov head, acceptance-confidence head). This is
an eager implementation, not vLLM's fused serving kernels.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class MarkovHead(nn.Module):
    def __init__(self, vocab, draft_vocab, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, draft_vocab, bias=False)

    def forward(self, previous):
        return self.markov_w2(self.markov_w1(previous))


class ConfidenceHead(nn.Module):
    def __init__(self, hidden_size, rank, with_markov):
        super().__init__()
        self.with_markov = with_markov
        self.proj = nn.Linear(hidden_size + (rank if with_markov else 0), 1, dtype=torch.float32)

    def forward(self, hidden, markov):
        x = torch.cat((hidden, markov), -1) if self.with_markov else hidden
        return torch.sigmoid(F.linear(x.float(), self.proj.weight.float(), self.proj.bias.float())).squeeze(-1)


def rotate(x, positions, theta):
    dim = x.shape[-1]
    freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angle = positions.float()[:, None] * freq[None, :]
    cos, sin = angle.cos()[:, None, :].to(x.dtype), angle.sin()[:, None, :].to(x.dtype)
    first, second = x.chunk(2, -1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, self.nq, self.nkv = c['hidden_size'], c['num_attention_heads'], c['num_key_value_heads']
        self.dim = c.get('head_dim', h // self.nq)
        self.theta = c.get('rope_parameters', {}).get('rope_theta', c.get('rope_theta', 1000000.))
        bias = c.get('attention_bias', False)
        self.q_proj = nn.Linear(h, self.nq * self.dim, bias=bias)
        self.k_proj = nn.Linear(h, self.nkv * self.dim, bias=bias)
        self.v_proj = nn.Linear(h, self.nkv * self.dim, bias=bias)
        self.o_proj = nn.Linear(self.nq * self.dim, h, bias=bias)
        self.q_norm = RMSNorm(self.dim, c['rms_norm_eps'])
        self.k_norm = RMSNorm(self.dim, c['rms_norm_eps'])

    def forward(self, query, context, start):
        pos = torch.arange(start, start + len(query), device=query.device)
        ctx_pos = torch.arange(len(context), device=query.device)
        q = rotate(self.q_norm(self.q_proj(query).view(-1, self.nq, self.dim)), pos, self.theta)
        k = rotate(self.k_norm(self.k_proj(query).view(-1, self.nkv, self.dim)), pos, self.theta)
        ck = rotate(self.k_norm(self.k_proj(context).view(-1, self.nkv, self.dim)), ctx_pos, self.theta)
        v = self.v_proj(query).view(-1, self.nkv, self.dim)
        cv = self.v_proj(context).view(-1, self.nkv, self.dim)
        k, v = torch.cat((ck, k)), torch.cat((cv, v))
        k = k.repeat_interleave(self.nq // self.nkv, dim=1)
        v = v.repeat_interleave(self.nq // self.nkv, dim=1)
        # Context is fixed; every query slot may read the whole noise block.
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), is_causal=False)
        return self.o_proj(out.squeeze(0).transpose(0, 1).reshape(len(query), -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c['hidden_size'], c['intermediate_size'], bias=False)
        self.up_proj = nn.Linear(c['hidden_size'], c['intermediate_size'], bias=False)
        self.down_proj = nn.Linear(c['intermediate_size'], c['hidden_size'], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c['hidden_size'], c['rms_norm_eps'])
        self.post_attention_layernorm = RMSNorm(c['hidden_size'], c['rms_norm_eps'])

    def forward(self, x, context, start):
        x = x + self.self_attn(self.input_layernorm(x), context, start)
        return x + self.mlp(self.post_attention_layernorm(x))


def normalize_config(raw):
    c = dict(raw.get('transformer_layer_config', raw))
    c.update({k: v for k, v in raw.items() if k != 'transformer_layer_config'})
    dc = c.get('dflash_config') or {}
    for key in ('mask_token_id', 'target_layer_ids'):
        if key not in c and key in dc:
            c[key] = dc[key]
    if 'aux_hidden_state_layer_ids' in raw:
        # Speculators stores hidden-state tuple indices, including embeddings.
        c['target_layer_ids'] = [i - 1 for i in raw['aux_hidden_state_layer_ids']]
    if 'eagle_aux_hidden_state_layer_ids' in raw:
        c['target_layer_ids'] = [i - 1 for i in raw['eagle_aux_hidden_state_layer_ids']]
    c.setdefault('sample_from_anchor', 'speculators_config' not in raw)
    if (c.get('markov_head_type', 'vanilla') != 'vanilla'
            or c.get('hidden_act', 'silu') != 'silu'
            or c.get('use_sliding_window') or dc.get('use_swa')
            or c.get('is_neox_style', True) is False
            or c.get('is_causal', False) or dc.get('causal', False)
            or any(t != 'full_attention' for t in c.get('layer_types', []))
            or c.get('rope_scaling')
            or c.get('rope_parameters', {}).get('rope_type', 'default') != 'default'
            or c.get('quantization_config') or c.get('dspark_draft_topk') is not None):
        raise ValueError('DSpark requires unquantized, full non-causal attention, default NeoX RoPE and vanilla Markov heads')
    if not c.get('target_layer_ids') or c.get('mask_token_id') is None:
        raise ValueError('DSpark checkpoint must specify target layers and mask_token_id')
    if not c.get('enable_confidence_head', False):
        raise ValueError('DSpark checkpoint must include a trained confidence head')
    return c


class DSparkModel(nn.Module):
    def __init__(self, config, target_hidden_size):
        super().__init__()
        self.config = normalize_config(config)
        c = self.config
        self.target_layer_ids = c['target_layer_ids']
        h, v, rank = c['hidden_size'], c['vocab_size'], c['markov_rank']
        draft_v = c.get('draft_vocab_size') or v
        if not 0 <= c['mask_token_id'] < v:
            raise ValueError('mask_token_id must fit the draft embedding vocabulary')
        self.embed_tokens = nn.Embedding(v, h)
        self.fc = nn.Linear(target_hidden_size * len(self.target_layer_ids), h, bias=False)
        self.hidden_norm = RMSNorm(h, c['rms_norm_eps'])
        self.layers = nn.ModuleList([Layer(c) for _ in range(c['num_hidden_layers'])])
        self.norm = RMSNorm(h, c['rms_norm_eps'])
        self.lm_head = nn.Linear(h, draft_v, bias=False)
        self.markov_head = MarkovHead(v, draft_v, rank)
        self.confidence_head = ConfidenceHead(h, rank, c.get('confidence_head_with_markov', False))
        self.register_buffer('d2t', torch.arange(draft_v), persistent=False)

    def project(self, aux):
        return self.fc(aux)

    def block(self, projected_context, ids, start):
        x = self.embed_tokens(ids)
        context = self.hidden_norm(projected_context)
        for layer in self.layers:
            x = layer(x, context, start)
        return self.norm(x)

    @classmethod
    def from_pretrained(cls, path, target, target_config):
        from safetensors import safe_open
        path = Path(path)
        c = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        with torch.device('cpu'):
            model = cls(c, target_config.hidden_size)
        if any(not isinstance(i, int) or i < 0 or i >= target_config.num_hidden_layers
               for i in model.target_layer_ids):
            raise ValueError('DSpark target layer ids are outside the target model')
        params = dict(model.named_parameters())
        loaded = set()
        files = sorted(path.glob('*.safetensors'))
        if not files:
            raise FileNotFoundError(f'No DSpark safetensors checkpoint in {path}')
        for file in files:
            with safe_open(file, framework='pt', device='cpu') as f:
                for original in f.keys():
                    name = original.removeprefix('model.')
                    if name in ('t2d', 'mask_embedding'):
                        continue
                    tensor = f.get_tensor(original)
                    if name in ('d2t', 'draft_id_to_target_id'):
                        # vLLM / Speculators stores offsets from draft token IDs.
                        model.d2t = tensor.long() + torch.arange(tensor.numel(), device='cpu')
                        loaded.add('d2t')
                        continue
                    if name not in params:
                        raise ValueError(f'Unsupported DSpark checkpoint tensor: {original}')
                    if params[name].shape != tensor.shape:
                        raise ValueError(f'DSpark tensor shape mismatch: {original}')
                    params[name].data.copy_(tensor)
                    loaded.add(name)
        # Checkpoint contracts allow shared target embeddings / full LM head.
        for name, source in (('embed_tokens.weight', target.model.embed_tokens.weight),
                             ('lm_head.weight', target.lm_head.weight)):
            if name not in loaded and params[name].shape == source.shape:
                params[name].data.copy_(source.detach().cpu())
                loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f'Incomplete DSpark checkpoint: {sorted(missing)}')
        if len(model.d2t) < target_config.vocab_size and 'd2t' not in loaded:
            raise ValueError('Reduced-vocabulary DSpark checkpoint requires d2t mapping')
        if (model.d2t.min() < 0 or model.d2t.max() >= target_config.vocab_size
                or len(model.d2t.unique()) != len(model.d2t)):
            raise ValueError('Invalid DSpark draft-to-target vocabulary mapping')
        return model.eval()
