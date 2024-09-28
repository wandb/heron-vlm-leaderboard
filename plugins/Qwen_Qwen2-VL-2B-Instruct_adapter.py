import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2Vl2BInstructAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
        'qwen-vl-utils',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map=device
        )
        min_pixels = 128 * 28 * 28
        max_pixels = 9999 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("Qwen/Qwen2-VL-2B-Instruct")

    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.get('max_length', 512),
                temperature=self.config.get('temperature', 0.7),
                do_sample=self.config.get('do_sample', True),
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return output_text[0]
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return ""

    async def verify(self) -> bool:
        try:
            test_question = "What is in this image?"
            test_image = "test.jpg"
            response = await self.generate_response(test_question, test_image)
            return len(response) > 0
        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_new_tokens": {
                "type": "integer",
                "default": 128,
                "description": "Maximum number of new tokens to generate"
            },
            "min_pixels": {
                "type": "integer",
                "default": 256 * 28 * 28,
                "description": "Minimum number of pixels for image resizing"
            },
            "max_pixels": {
                "type": "integer",
                "default": 1280 * 28 * 28,
                "description": "Maximum number of pixels for image resizing"
            }
        }

def register_plugin(manager):
    manager.register_adapter("qwen2-vl-2b-instruct", Qwen2Vl2BInstructAdapter)