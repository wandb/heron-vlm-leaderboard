import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoProcessor, LlavaForConditionalGeneration
import requests

class Llava157BHfAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.35.3',
        'pillow>=8.0.0',
        'requests>=2.25.0'
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("llava-hf/llava-1.5")

    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device, torch.float16)
            
            output = self.model.generate(
                **inputs, 
                    max_new_tokens=self.config.get('max_length', 512),
                    temperature=self.config.get('temperature', 0.7),
                    do_sample=self.config.get('do_sample', True),
                    #no_repeat_ngram_size=self.config.get('no_repeat_ngram_size', 2),
                    )
            response = self.processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[-1]
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def verify(self) -> bool:
        try:
            test_image_path = "test.jpg"
            test_question = "What is in this image?"
            response = await self.generate_response(test_question, test_image_path)
            return len(response) > 0
        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_new_tokens": {
                "type": "integer",
                "default": 512,
                "description": "Maximum number of new tokens to generate"
            },
            "do_sample": {
                "type": "boolean",
                "default": False,
                "description": "Whether to use sampling for generation"
            }
        }

def register_plugin(manager):
    manager.register_adapter("llava-1.5-7b-hf", Llava157BHfAdapter)