import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import requests
import logging

class Molmo72B0924Adapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
        'einops',
        'torchvision',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # 初期化時にbfloat16を指定
            device_map='auto'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # 初期化時にbfloat16を指定
            device_map='auto'
        )
        self.model.eval()

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("allenai/Molmo-72B-0924")

    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor.process(
                images=[image],
                text=question
            )
            # テンソルのデバイス移動のみ行う（dtype変換は行わない）
            inputs = {k: v.to(self.model.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(
                        max_new_tokens=self.config.get('max_length', 512),
                        stop_strings="<|endoftext|>",
                        temperature=self.config.get('temperature', 0.7),
                    ),
                    tokenizer=self.processor.tokenizer
                )
            
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "An error occurred while generating the response."

    async def verify(self) -> bool:
        try:
            test_image_path = "test.jpg"
            test_question = "Describe this image."
            response = await self.generate_response(test_question, test_image_path)
            return len(response) > 0
        except Exception as e:
            logging.error(f"Verification failed: {str(e)}")
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_new_tokens": {
                "type": "integer",
                "default": 200,
                "description": "Maximum number of new tokens to generate"
            },
            "temperature": {
                "type": "number",
                "default": 1.0,
                "description": "Sampling temperature"
            }
        }

def register_plugin(manager):
    manager.register_adapter("molmo-72b-0924", Molmo72B0924Adapter)