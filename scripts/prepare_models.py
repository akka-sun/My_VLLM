"""Download pinned model snapshots without importing CUDA inference code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_MODELS = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-VL-2B-Instruct']


def verify_snapshot(path):
    path = Path(path)
    if not (path / 'config.json').is_file():
        raise ValueError('missing config.json')
    if not (path / 'tokenizer.json').is_file():
        raise ValueError('missing tokenizer.json')
    index = path / 'model.safetensors.index.json'
    shards = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not shards or any(not (path / name).is_file() for name in shards):
        raise ValueError('missing safetensors weights/shards')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='repeat to select multiple repositories')
    parser.add_argument('--revision', default='main', help='resolved to immutable SHA before download')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'models')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    api = HfApi()
    for model in args.model or DEFAULT_MODELS:
        revision = api.model_info(model, revision=args.revision).sha
        destination = args.output / model.replace('/', '--')
        metadata = destination / 'preparation.json'
        if metadata.exists():
            previous = json.loads(metadata.read_text(encoding='utf-8'))
            if previous['revision'] != revision:
                parser.error(f'{destination} contains another revision; select a different --output')
            if not previous['metadata_only']:
                verify_snapshot(destination)
                print(json.dumps(previous), flush=True)
                continue
        snapshot_download(model, revision=revision, local_dir=destination,
                          allow_patterns=['*.json', '*.txt', '*.jinja', '*.model'] +
                          ([] if args.metadata_only else ['*.safetensors']), max_workers=4)
        if not args.metadata_only:
            verify_snapshot(destination)
        record = {'repo_id': model, 'revision': revision, 'metadata_only': args.metadata_only,
                  'downloaded_at': datetime.now(timezone.utc).isoformat()}
        metadata.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
