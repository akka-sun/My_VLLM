import pytest
import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence


def sample(logits, top_k=None, top_p=None, temperature=1.0):
    # Exercise the tensor implementation eagerly; Windows lacks Inductor/Triton.
    return Sampler.forward.__wrapped__(
        Sampler(), logits, torch.full((len(logits),), temperature, device=logits.device),
        top_k, top_p,
    )


@pytest.mark.parametrize("kwargs", [
    {"top_k": -1}, {"top_k": 1.5}, {"top_k": True},
    {"top_p": 0}, {"top_p": 1.1}, {"top_p": float("nan")},
])
def test_invalid_filter_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_filters_follow_request_into_sequence():
    seq = Sequence([1], SamplingParams(top_k=3, top_p=0.8))
    assert (seq.top_k, seq.top_p) == (3, 0.8)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_batch_filters_and_nucleus_crossing(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    # Rows: top-1, top-2 then nucleus, unfiltered, nucleus crossing at second token.
    probs = torch.tensor([[0.6, 0.3, 0.1]] * 4096, device=device)
    logits = probs.log()
    top_k = torch.tensor([1, 2, 0, 0] * 1024, device=device)
    top_p = torch.tensor([1.0, 0.6, 1.0, 0.7] * 1024, device=device)
    tokens = sample(logits, top_k, top_p).reshape(-1, 4)
    assert set(tokens[:, 0].tolist()) == {0}
    # After top-2 renormalization, token 0 alone already exceeds 0.6.
    assert set(tokens[:, 1].tolist()) == {0}
    assert set(tokens[:, 2].tolist()) == {0, 1, 2}
    # 0.6 < 0.7, so include the second token which crosses the threshold.
    assert set(tokens[:, 3].tolist()) == {0, 1}
    assert abs((tokens[:, 3] == 0).float().mean().item() - 2 / 3) < 0.06


def test_default_filters_preserve_sampling_and_do_not_mutate_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0]] * 256)
    original = logits.clone()
    torch.manual_seed(123)
    expected = sample(logits.clone(), temperature=0.6)
    torch.manual_seed(123)
    actual = sample(logits, torch.zeros(256, dtype=torch.long), torch.ones(256), 0.6)
    assert torch.equal(actual, expected)
    assert torch.equal(logits, original)


def test_top_k_exceeding_vocabulary_and_tiny_top_p():
    logits = torch.tensor([[0.0, 1.0, 2.0]] * 32)
    assert set(sample(logits, torch.full((32,), 99), torch.full((32,), 1e-8)).tolist()) == {2}


def test_top_k_keeps_exact_count_with_tied_logits():
    logits = torch.zeros(2048, 3)
    tokens = sample(logits, torch.ones(2048, dtype=torch.long), torch.ones(2048))
    assert len(set(tokens.tolist())) == 1


def test_nucleus_exact_boundary_keeps_shortest_prefix():
    logits = torch.tensor([[0.5, 0.25, 0.25]] * 512).log()
    assert set(sample(logits, top_p=torch.full((512,), 0.5)).tolist()) == {0}


def test_tiny_valid_probability_still_keeps_highest_token_after_float32_conversion():
    p = SamplingParams(top_p=1e-50).top_p
    logits = torch.tensor([[0.0, 1.0, 2.0]] * 32)
    assert set(sample(logits, top_p=torch.full((32,), p)).tolist()) == {2}


def test_sampler_captures_as_one_torch_compile_graph():
    import subprocess
    import sys
    from pathlib import Path

    # Isolate from installed legacy Transformers: its file_utils module breaks
    # Torch 2.3 Dynamo's unrelated ModelOutput type check in this environment.
    script = """
import runpy
import torch
Sampler = runpy.run_path('nanovllm/layers/sampler.py')['Sampler']
compiled = torch.compile(Sampler.forward.__wrapped__, backend='eager', fullgraph=True)
tokens = compiled(Sampler(), torch.tensor([[0., 2., 1.]]),
                  torch.ones(1), torch.ones(1, dtype=torch.long), torch.ones(1))
assert tokens.tolist() == [1], tokens
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_model_runner_transfers_request_filters_and_uses_them(monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path

    # Import the actual runner, replacing only the unavailable attention/model stack.
    model = types.ModuleType("nanovllm.models.qwen3")
    model.Qwen3ForCausalLM = object
    monkeypatch.setitem(sys.modules, "nanovllm.models.qwen3", model)
    spec = importlib.util.spec_from_file_location(
        "tested_model_runner", Path(__file__).resolve().parents[1] / "nanovllm/engine/model_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = object.__new__(module.ModelRunner)
    runner.rank = 0
    seqs = [Sequence([1], SamplingParams(top_k=1)),
            Sequence([2], SamplingParams(top_p=0.1))]
    temperatures, ks, ps = runner.prepare_sample(seqs)
    assert ks.tolist() == [1, 0]
    assert torch.allclose(ps, torch.tensor([1.0, 0.1], device="cuda"))
    assert temperatures.is_cuda and ks.is_cuda and ps.is_cuda
    runner.prepare_decode = lambda seqs: (None, None)
    runner.run_model = lambda *args: torch.tensor([[0.0, 2.0, 1.0]] * 2, device="cuda")
    runner.sampler = lambda *args: Sampler.forward.__wrapped__(Sampler(), *args)
    assert runner.run(seqs, False) == [1, 1]
    default_seqs = [Sequence([1])]
    assert runner.prepare_sample(default_seqs)[1:] == (None, None)
    huge_k_seqs = [Sequence([1], SamplingParams(top_k=2**100))]
    _, huge_ks, _ = runner.prepare_sample(huge_k_seqs)
    assert huge_ks.item() >= 3
