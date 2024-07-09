import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenVLChatAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
        'einops',
        'transformers_stream_generator>=0.0.4',
        'torchvision',
        'tiktoken',
        'matplotlib',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith('Qwen/Qwen-VL-Chat')

    @torch.inference_mode()
    async def generate_response(self, question: str, image_path: str) -> str:
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': question},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    async def verify(self) -> bool:
        try:
            test_question = "What is in this image?"
            test_image_path = "test.jpg"
            response = await self.generate_response(test_question, test_image_path)
            return len(response) > 0
        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_length": int,
            "temperature": float,
        }

def register_plugin(manager):
    manager.register_adapter("qwen_vl_chat", QwenVLChatAdapter)