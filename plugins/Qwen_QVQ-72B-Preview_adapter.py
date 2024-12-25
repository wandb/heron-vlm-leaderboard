import logging
import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class Qvq72BPreviewAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
        'qwen-vl-utils>=0.1.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if "16" in device else "auto",
                device_map="auto"
            )
            self.model.eval()
        except Exception as e:
            logging.error(f"Failed to initialize QVQ model: {e}")

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("Qwen/QVQ-72B-Preview")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        max_pixels = 1280 * 28 * 28
        current_pixels = image.width * image.height
        if current_pixels > max_pixels:
            scale = (max_pixels / current_pixels) ** 0.5
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            # アスペクト比を維持したままリサイズ
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "あなたは親切で無害なアシスタントです。あなたはアリババが開発したQwenです。ステップバイステップで考えて答えてください。使用言語は必ず日本語でお願いします。"
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=[image],
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return output_text[0].strip()
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "An error occurred during generation."

    async def verify(self) -> bool:
        try:
            result = await self.generate_response("What is in this picture?", "test.jpg")
            if result and len(result) > 0:
                return True
        except Exception as e:
            logging.error(f"Verification failed: {e}")
        return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_new_tokens": {
                "type": "integer",
                "default": 8192,
                "description": "Maximum number of tokens to generate in a single response."
            }
        }

def register_plugin(manager):
    manager.register_adapter("qvq-72b-preview", Qvq72BPreviewAdapter)