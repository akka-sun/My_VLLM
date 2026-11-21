"""Local Qwen3-VL reference adapter using Transformers generation."""
from pathlib import Path

import torch


class Qwen3VLAdapter:
    def __init__(self, model: str | Path, device: str = 'cuda:0'):
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

        model = Path(model)
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        if config.model_type != 'qwen3_vl':
            raise ValueError('Qwen3VLAdapter requires a local Qwen3-VL checkpoint')
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model, local_files_only=True, dtype=config.text_config.dtype,
            attn_implementation='sdpa',
        ).to(self.device).eval()

    def prepare_image(self, image: str | Path, prompt: str):
        from PIL import Image

        with Image.open(image) as source:
            pixels = source.convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': pixels}, {'type': 'text', 'text': prompt},
        ]}]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt',
        ).to(self.device)

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens: int):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        tokens = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        answer = tokens[0, inputs.input_ids.shape[1]:]
        return {'text': self.processor.decode(answer, skip_special_tokens=True),
                'output_tokens': answer.numel()}
