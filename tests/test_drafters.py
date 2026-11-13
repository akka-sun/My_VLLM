from types import SimpleNamespace

import torch
import pytest

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_independent_drafter_crops_rejected_suffix_and_releases_state(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from nanovllm.engine.drafters import IndependentDrafter
    # A real, tiny local Qwen3 model; no downloaded weights or network calls.
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(4)
    hf = Qwen3Config(vocab_size=12, hidden_size=8, intermediate_size=16,
                     num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=4, max_position_embeddings=64)
    model = Qwen3ForCausalLM(hf).to(device).eval()
    drafter = IndependentDrafter(model)
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0, max_tokens=8))
    ids, probs, _ = drafter.propose(seq, 3, 1)
    assert len(ids) == 3 and probs.shape == (3, 12)
    # Reject the first draft and commit a different token.
    seq.append_token((ids[0] + 1) % 12)
    ids2, _, _ = drafter.propose(seq, 2, 1)
    direct = model(torch.tensor([seq.token_ids], device=device)).logits[0, -1].argmax().item()
    assert ids2[0] == direct
    drafter.retain(set())
    assert not drafter.caches


def test_dspark_proposal_markov_context_capture_and_layouts():
    from nanovllm.engine.drafters import DSparkDrafter
    from nanovllm.models.dspark import DSparkModel
    cfg = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
               num_attention_heads=2, num_key_value_heads=1, head_dim=4,
               vocab_size=12, markov_rank=2, target_layer_ids=[0], mask_token_id=0,
               enable_confidence_head=True, confidence_head_with_markov=True,
               rms_norm_eps=1e-6, rope_theta=10000., block_size=4)
    engine = SimpleNamespace(max_num_seqs=2, max_model_len=32, max_num_batched_tokens=16,
                             speculative_config={"num_speculative_tokens": 3})
    for anchor in (True, False):
        model = DSparkModel(cfg | {"sample_from_anchor": anchor}, 8).eval()
        drafter = DSparkDrafter(model, engine)
        seq = Sequence([1, 2, 3], SamplingParams(temperature=0, max_tokens=8))
        seq.num_computed_tokens = 2
        drafter.capture(seq, 0, torch.randn(2, 8))
        ids, probs, confidence = drafter.propose(seq, 3, 1)
        assert len(ids) == 3 and probs.shape == (3, 12)
        assert confidence.shape == (3,) and torch.all((confidence > 0) & (confidence < 1))
        assert probs.argmax(-1).tolist() == ids
        drafter.retain(set())
        assert not drafter.slots


def test_dspark_checkpoint_roundtrip_and_missing_head_rejected(tmp_path):
    import json
    import pytest
    from safetensors.torch import save_file
    from nanovllm.models.dspark import DSparkModel
    cfg = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
               num_attention_heads=2, num_key_value_heads=1, head_dim=4,
               vocab_size=12, markov_rank=2, target_layer_ids=[0], mask_token_id=0,
               enable_confidence_head=True, rms_norm_eps=1e-6)
    model = DSparkModel(cfg, 8)
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=model.embed_tokens), lm_head=model.lm_head)
    target_config = SimpleNamespace(hidden_size=8, num_hidden_layers=2, vocab_size=12)
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    save_file(model.state_dict() | {'mask_embedding': torch.zeros(8)}, tmp_path / 'model.safetensors')
    loaded = DSparkModel.from_pretrained(tmp_path, target, target_config)
    context, ids = torch.randn(2, 8), torch.tensor([1, 0, 0])
    assert torch.equal(model.block(context, ids, 2), loaded.block(context, ids, 2))
    state = model.state_dict()
    state.pop('confidence_head.proj.bias')
    save_file(state, tmp_path / 'model.safetensors')
    with pytest.raises(ValueError, match='Incomplete'):
        DSparkModel.from_pretrained(tmp_path, target, target_config)


def test_dspark_verification_budget_is_independent_of_random_proposals():
    """Token-dependent truncation would select q and bias the output law."""
    from nanovllm.models.dspark import DSparkModel
    from nanovllm.engine.drafters import DSparkDrafter
    cfg = dict(hidden_size=2, intermediate_size=4, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1, head_dim=2, vocab_size=2,
        markov_rank=1, target_layer_ids=[0], mask_token_id=0,
        enable_confidence_head=True, confidence_head_with_markov=True, rms_norm_eps=1e-6)
    model = DSparkModel(cfg, 2)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.markov_head.markov_w2.weight.zero_()
        model.markov_head.markov_w1.weight.copy_(torch.tensor([[10.], [-10.]]))
        model.confidence_head.proj.weight.copy_(torch.tensor([[0., 0., 1.]]))
        model.confidence_head.proj.bias.zero_()
    config = SimpleNamespace(max_num_seqs=1, max_model_len=8,
        max_num_batched_tokens=8, speculative_config={'num_speculative_tokens': 2})
    drafter = DSparkDrafter(model, config)
    seq = Sequence([1, 0], SamplingParams(max_tokens=4))
    seq.num_computed_tokens = 1
    drafter.capture(seq, 0, torch.zeros(1, 2))
    samples, confidence = [], []
    for seed in (1, 2, 3, 4):
        torch.manual_seed(seed)
        ids, _, scores = drafter.propose(seq, 2, 1)
        samples.append(ids)
        confidence.append(scores)
    assert len({tuple(ids) for ids in samples}) > 1
    assert all(torch.equal(c, confidence[0]) for c in confidence)


def test_reduced_vocab_speculators_checkpoint_uses_target_ids(tmp_path):
    import json
    from safetensors.torch import save_file
    from nanovllm.models.dspark import DSparkModel
    from nanovllm.engine.drafters import DSparkDrafter
    layer = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=12,
        rms_norm_eps=1e-6)
    cfg = dict(transformer_layer_config=layer, aux_hidden_state_layer_ids=[1],
        speculators_config={'algorithm': 'dspark'}, draft_vocab_size=4,
        markov_rank=2, mask_token_id=0, enable_confidence_head=True, block_size=4)
    model = DSparkModel(cfg, 8).eval()
    state = model.state_dict() | {'d2t': torch.tensor([0, 2, 3, 5]), 't2d': torch.zeros(12)}
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    save_file(state, tmp_path / 'model.safetensors')
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=model.embed_tokens), lm_head=model.lm_head)
    target_config = SimpleNamespace(hidden_size=8, num_hidden_layers=2, vocab_size=12)
    loaded = DSparkModel.from_pretrained(tmp_path, target, target_config)
    assert loaded.d2t.tolist() == [0, 3, 5, 8]
    assert not loaded.config['sample_from_anchor'] and loaded.target_layer_ids == [0]
    config = SimpleNamespace(max_num_seqs=1, max_model_len=16, max_num_batched_tokens=8,
                             speculative_config={'num_speculative_tokens': 3})
    drafter = DSparkDrafter(loaded, config)
    seq = Sequence([1, 2], SamplingParams(max_tokens=8))
    seq.num_computed_tokens = 1
    drafter.capture(seq, 0, torch.randn(1, 8))
    ids, q, _ = drafter.propose(seq, 3, 1)
    assert set(ids) <= {0, 3, 5, 8}
    assert torch.equal(q[:, [1, 2, 4, 6, 7, 9, 10, 11]], torch.zeros(3, 8))
    state.pop('d2t')
    save_file(state, tmp_path / 'model.safetensors')
    with pytest.raises(ValueError, match='requires d2t'):
        DSparkModel.from_pretrained(tmp_path, target, target_config)
