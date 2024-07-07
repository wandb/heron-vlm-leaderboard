import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter

class CyberagentLlavaCalmSiglipAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("cyberagent/llava-calm2-siglip")

    async def generate_response(self, question: str, image_path: str) -> str:
        image = Image.open(image_path)
        
        prompt = f"""USER: <image>
{question}
ASSISTANT: """
        
        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.bfloat16)
            
            generate_ids = self.model.generate(
                **inputs,
                max_length=self.config.get('max_length', 256),
                do_sample=self.config.get('do_sample', False),
                temperature=self.config.get('temperature', 0.7),
                no_repeat_ngram_size=self.config.get('no_repeat_ngram_size', 3),
            )
            
            output = self.processor.tokenizer.decode(generate_ids[0][:-1], clean_up_tokenization_spaces=False)
            response = output.split("ASSISTANT: ")[1]
            return response
        
        except Exception as e:
            print(f"Error during model generation: {str(e)}")
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
            print(f"Verification failed: {str(e)}")
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
    manager.register_adapter("cyberagent_llava_calm2_siglip", CyberagentLlavaCalmSiglipAdapter)