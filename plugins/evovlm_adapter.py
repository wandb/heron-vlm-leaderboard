import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoModelForVision2Seq, AutoProcessor
import logging

class EvoVLMAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith('SakanaAI/EvoVLM-JP')

    @torch.inference_mode()
    async def generate_response(self, question: str, image_path: str) -> str:
        image = Image.open(image_path)
        
        messages = [
            {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。"},
            {"role": "user", "content": f"<image>\n{question}"},
        ]
        
        try:
            inputs = self.processor.image_processor(images=image, return_tensors="pt")
            inputs["input_ids"] = self.processor.tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            )
            
            output_ids = self.model.generate(
                **inputs.to(self.device),
                max_length=self.config.get('max_length', 256),
                do_sample=self.config.get('do_sample', True),
                temperature=self.config.get('temperature', 0.7),
                no_repeat_ngram_size=self.config.get('no_repeat_ngram_size', 3),
            )
            output_ids = output_ids[:, inputs.input_ids.shape[1]:]
            generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return generated_text
        
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            raise e
        finally:
            del inputs
            torch.cuda.empty_cache()

    async def verify(self) -> bool:
        try:
            test_question = "What is in this image?"
            test_image_path = "test.jpg"
            response = await self.generate_response(test_question, test_image_path)
            return len(response) > 0
        except Exception as e:
            logging.error(f"Verification failed: {str(e)}")
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_length": int,
            "do_sample": bool,
            "temperature": float,
            "no_repeat_ngram_size": int,
        }

def register_plugin(manager):
    manager.register_adapter("evovlm", EvoVLMAdapter)