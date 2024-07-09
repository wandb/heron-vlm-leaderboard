from plugins.base_adapter import BaseAdapter
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Dict, Any

class LLaVAAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("liuhaotian/llava-")

    async def generate_response(self, question: str, image_path: str) -> str:
        image = Image.open(image_path)
        inputs = self.processor(text=question, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config['max_length'],
                do_sample=True,
                temperature=self.config['temperature']
            )
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    async def verify(self) -> bool:
        try:
            test_question = "What is in this image?"
            test_image_path = "path/to/test/image.jpg"
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
    manager.register_adapter("llava", LLaVAAdapter)