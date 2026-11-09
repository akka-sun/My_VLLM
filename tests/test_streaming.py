from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


class Tokenizer:
    def encode(self, text):
        return [1]

    def decode(self, ids):
        return "".join({7: "A", 8: "B", 9: "!"}[i] for i in ids)


class Runner:
    def call(self, method, seqs, is_prefill):
        assert method == "run"
        return [7 if seq.num_completion_tokens == 0 else 9 for seq in seqs]


def engine(budget=8):
    result = object.__new__(LLMEngine)
    result.tokenizer = Tokenizer()
    result.model_runner = Runner()
    result.scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=budget, eos=9,
        kvcache_block_size=256, num_kvcache_blocks=16,
    ))
    return result


def test_step_returns_delta_before_finish_and_final_eos_once():
    llm = engine()
    llm.add_request([1], SamplingParams())
    first, _ = llm.step(stream=True)
    assert len(first) == 1
    assert first[0]["token_ids"] == [7]
    assert first[0]["finished"] is False
    last, _ = llm.step(stream=True)
    assert last == [dict(seq_id=first[0]["seq_id"], token_ids=[9], finished=True)]
    assert llm.is_finished()


def test_chunked_prefill_emits_nothing_until_token_is_appended():
    llm = engine(budget=2)
    llm.add_request([1, 2, 3], SamplingParams(max_tokens=1))
    assert llm.step(stream=True) == ([], 2)
    output, _ = llm.step(stream=True)
    assert output[0]["token_ids"] == [7]
    assert output[0]["finished"] is True


def test_stream_is_lazy_and_yields_each_request_incrementally():
    llm = engine()
    stream = llm.generate_stream([[1], [2]], SamplingParams())
    assert llm.is_finished()
    first = next(stream)
    assert first["text"] == "A"
    assert first["token_ids"] == [7]
    assert first["finished"] is False
    rest = list(stream)
    assert [x["text"] for x in rest] == ["A", "A!", "A!"]
    assert [x["token_ids"] for x in rest] == [[7], [9], [9]]
    assert len({x["seq_id"] for x in [first, *rest]}) == 2
    assert llm.is_finished()


def test_generate_keeps_existing_complete_output_contract():
    assert engine().generate([[1]], SamplingParams(), use_tqdm=False) == [
        {"text": "A!", "token_ids": [7, 9]}
    ]


def test_stream_rejects_mismatched_params_before_queuing():
    llm = engine()
    with pytest.raises(ValueError):
        next(llm.generate_stream([[1], [2]], [SamplingParams()]))
    assert llm.is_finished()
