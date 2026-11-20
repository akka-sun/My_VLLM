import importlib

import pytest


def test_workloads_are_seeded_and_do_not_change_global_rng():
    import random
    module = importlib.import_module('benchmarks.workloads')
    random.seed(91)
    before = random.getstate()
    a = module.make_workload('mixed', 20, 42)
    assert random.getstate() == before
    assert a == module.make_workload('mixed', 20, 42)
    assert a != module.make_workload('mixed', 20, 43)
    assert len({x['id'] for x in a}) == 20
    assert {x['scenario'] for x in a} == {'short', 'long'}
    assert all(x['messages'][0]['role'] == 'user' for x in a)


@pytest.mark.parametrize('preset,count', [('bad', 1), ('short', 0), ('short', -1)])
def test_workloads_reject_invalid_input(preset, count):
    module = importlib.import_module('benchmarks.workloads')
    with pytest.raises(ValueError):
        module.make_workload(preset, count, 42)


def test_model_preparation_checks_required_files(tmp_path):
    module = importlib.import_module('scripts.prepare_models')
    (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='tokenizer'):
        module.verify_snapshot(tmp_path)
    (tmp_path / 'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError, match='safetensors'):
        module.verify_snapshot(tmp_path)
    (tmp_path / 'model.safetensors').write_bytes(b'test fixture')
    module.verify_snapshot(tmp_path)
