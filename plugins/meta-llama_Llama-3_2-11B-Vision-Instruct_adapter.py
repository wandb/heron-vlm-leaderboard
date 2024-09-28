import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import MllamaForConditionalGeneration, AutoProcessor

class Llama3211BVisionInstructAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.45.0',
        'pillow>=8.0.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("meta-llama/Llama-3.2-11B-Vision-Instruct")

    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
            
            output = self.model.generate(**inputs, max_new_tokens=self.config.get("max_length", 512))
            return self.processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "An error occurred while generating the response."

    async def verify(self) -> bool:
        try:
            test_question = "What do you see in this image?"
            test_image_path = "test.jpg"
            response = await self.generate_response(test_question, test_image_path)
            return len(response) > 0
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_new_tokens": {
                "type": "integer",
                "default": 100,
                "description": "Maximum number of new tokens to generate"
            }
        }

def register_plugin(manager):
    manager.register_adapter("llama-3.2-11b-vision-instruct", Llama3211BVisionInstructAdapter)