from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def test_speculative_probability_filters_and_greedy():
    from nanovllm.engine.speculative import probabilities
    logits = torch.tensor([[0.6, 0.3, 0.1]]).log()
    assert torch.equal(probabilities(logits, 0), torch.tensor([[1., 0., 0.]]))
    assert torch.allclose(probabilities(logits, 1, 2, .7), torch.tensor([[2/3, 1/3, 0.]]))
    assert torch.equal(logits, torch.tensor([[0.6, 0.3, 0.1]]).log())


def test_verify_all_accepted_includes_bonus_and_first_rejection_corrects():
    from nanovllm.engine.speculative import verify
    p = torch.eye(3)[[1, 2, 0]]
    assert verify([1, 2], torch.eye(3)[[1, 2]], p) == [1, 2, 0]
    assert verify([0, 2], torch.eye(3)[[0, 2]], p) == [1]


def test_stochastic_rejection_preserves_target_distribution():
    from nanovllm.engine.speculative import verify
    torch.manual_seed(17)
    q = torch.tensor([[.85, .15]])
    p = torch.tensor([[.2, .8], [.3, .7]])
    counts = torch.zeros(2)
    for _ in range(5000):
        token = torch.multinomial(q[0], 1).item()
        counts[verify([token], q, p)[0]] += 1
    assert torch.allclose(counts / counts.sum(), p[0], atol=.025)


def test_scheduler_reserves_lookahead_and_commits_only_valid_prefix(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 2)
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.speculative import SpeculativeResult
    cfg = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=8, eos=9,
                          kvcache_block_size=2, num_kvcache_blocks=8,
                          speculative_config={"method": "draft_model", "num_speculative_tokens": 3})
    scheduler = Scheduler(cfg)
    seq = Sequence([1, 2], SamplingParams(max_tokens=8))
    scheduler.add(seq)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [3], prefill)
    batch, prefill = scheduler.schedule()
    assert seq.num_speculative_tokens == 3
    assert seq.num_scheduled_tokens == 4
    assert len(seq.block_table) == 3
    scheduler.postprocess(batch, [SpeculativeResult([4, 5])], prefill)
    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.num_computed_tokens == 4
    assert len(seq.block_table) == 2
    assert scheduler.block_manager.get_computed_blocks(Sequence([1, 2, 3, 4, 8])) == seq.block_table
    assert all(block.token_ids != [5, 0] for block in scheduler.block_manager.blocks)


def test_scheduler_stops_inside_verified_block_and_releases_pages(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 2)
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.speculative import SpeculativeResult
    cfg = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=8, eos=9,
                          kvcache_block_size=2, num_kvcache_blocks=8,
                          speculative_config={"method": "draft_model", "num_speculative_tokens": 3})
    scheduler = Scheduler(cfg)
    seq = Sequence([1], SamplingParams(max_tokens=8))
    scheduler.add(seq)
    batch, flag = scheduler.schedule()
    scheduler.postprocess(batch, [2], flag)
    batch, flag = scheduler.schedule()
    scheduler.postprocess(batch, [SpeculativeResult([3, 9, 7])], flag)
    assert seq.completion_token_ids == [2, 3, 9]
    assert seq.is_finished and not seq.block_table
    assert len(scheduler.block_manager.free_block_ids) == 8


def test_confidence_length_uses_prefix_survival_and_cost():
    from nanovllm.engine.speculative import verification_length
    assert verification_length(torch.tensor([.99, .99, .01]), [1., 1.1, 1.2, 10.]) == 2
    assert verification_length(torch.tensor([.99, .99]), [1., 100., 200.]) == 0


def test_greedy_sampling_params_supported():
    assert SamplingParams(temperature=0).temperature == 0


@pytest.mark.parametrize("options", [
    {"method": "unknown", "model": "x", "num_speculative_tokens": 3},
    {"method": "draft_model", "model": "x", "num_speculative_tokens": 0},
    {"method": "draft_model", "num_speculative_tokens": 3},
    {"method": "dspark", "model": "x", "num_speculative_tokens": True},
])
def test_invalid_speculative_options(options):
    from nanovllm.config import validate_speculative_config
    with pytest.raises(ValueError):
        validate_speculative_config(options)


def test_config_enables_eager_and_normalizes_transformers_dtype(tmp_path):
    from transformers import Qwen3Config
    from nanovllm.config import Config
    Qwen3Config(torch_dtype='float16').save_pretrained(tmp_path)
    options = dict(method='draft_model', model=str(tmp_path), num_speculative_tokens=3)
    config = Config(str(tmp_path), speculative_config=options)
    assert config.enforce_eager and config.hf_config.dtype == torch.float16
    assert config.speculative_config == options and config.speculative_config is not options
    with pytest.raises(ValueError, match='tensor_parallel_size=1'):
        Config(str(tmp_path), speculative_config=options, tensor_parallel_size=2)


def test_dspark_markov_and_confidence_depend_on_previous_token():
    from nanovllm.models.dspark import MarkovHead, ConfidenceHead
    markov = MarkovHead(3, 3, 2)
    with torch.no_grad():
        markov.markov_w1.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
        markov.markov_w2.weight.copy_(torch.tensor([[1., 0.], [0., 2.], [0., 0.]]))
    assert torch.equal(markov(torch.tensor(0)), torch.tensor([1., 0., 0.]))
    assert torch.equal(markov(torch.tensor(1)), torch.tensor([0., 2., 0.]))
    confidence = ConfidenceHead(2, 2, True)
    with torch.no_grad():
        confidence.proj.weight.copy_(torch.tensor([[0., 0., 1., -1.]]))
        confidence.proj.bias.zero_()
    assert confidence(torch.zeros(2), markov.markov_w1(torch.tensor(0))) > .5
    assert confidence(torch.zeros(2), markov.markov_w1(torch.tensor(1))) < .5


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_dspark_parallel_backbone_reads_context_and_noncausal_block(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from nanovllm.models.dspark import DSparkModel
    cfg = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
               num_attention_heads=2, num_key_value_heads=1, head_dim=4,
               vocab_size=12, markov_rank=2, target_layer_ids=[0],
               mask_token_id=0, enable_confidence_head=True,
               rms_norm_eps=1e-6, rope_theta=10000.)
    torch.manual_seed(12)
    model = DSparkModel(cfg, target_hidden_size=8).to(device).eval()
    context = torch.randn(3, 8, device=device)
    ids = torch.tensor([1, 0, 0], device=device)
    hidden = model.block(context, ids, 3)
    assert hidden.shape == (3, 8)
    assert not torch.allclose(hidden, model.block(context + 1, ids, 3))
    changed = ids.clone()
    changed[-1] = 2
    assert not torch.allclose(hidden[0], model.block(context, changed, 3)[0])


@pytest.mark.parametrize('method', ['draft_model', 'dspark'])
def test_speculative_page_pressure_preemption_and_output_limits(monkeypatch, method):
    import random
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.speculative import SpeculativeResult
    monkeypatch.setattr(Sequence, 'block_size', 4)
    rng = random.Random(17)
    config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=7, eos=99,
        kvcache_block_size=4, num_kvcache_blocks=4,
        speculative_config={'method': method, 'num_speculative_tokens': 3})
    scheduler = Scheduler(config)
    requests = [Sequence([1] * n, SamplingParams(max_tokens=5)) for n in (5, 5, 7, 3, 6)]
    for seq in requests:
        scheduler.add(seq)
    rounds = 0
    while not scheduler.is_finished():
        batch, flag = scheduler.schedule()
        assert sum(s.num_scheduled_tokens for s in batch) <= 7
        outputs = []
        for seq in batch:
            if seq.num_speculative_tokens:
                committed = rng.randint(1, seq.num_speculative_tokens + 1)
                outputs.append(SpeculativeResult(list(range(
                    10 + seq.num_completion_tokens, 10 + seq.num_completion_tokens + committed))))
            else:
                outputs.append(10 + seq.num_completion_tokens)
        scheduler.postprocess(batch, outputs, flag)
        for seq in requests:
            assert seq.num_completion_tokens <= seq.max_tokens
            if not seq.is_finished:
                assert seq.num_computed_tokens <= seq.num_tokens
            assert seq.completion_token_ids == list(range(10, 10 + seq.num_completion_tokens))
        rounds += 1
        assert rounds < 100
    assert scheduler.num_preemptions > 0
    assert all(s.completion_token_ids == [10, 11, 12, 13, 14] for s in requests)
    assert len(scheduler.block_manager.free_block_ids) == 4
