import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoModelForCausalLM, AutoProcessor
import logging

class Phi3Vision128KInstructAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager',
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name == 'microsoft/Phi-3-vision-128k-instruct'

    @torch.inference_mode()
    async def generate_response(self, question: str, image_path: str) -> str:
        image = Image.open(image_path)
        
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{question}"},
        ]
        
        try:
            prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)
            
            generation_args = {
                "max_new_tokens": self.config.get('max_length', 256),
                "temperature": self.config.get('temperature', 0.7),
                "do_sample": self.config.get('do_sample', True),
            }
            
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id, 
                no_repeat_ngram_size=self.config.get('no_repeat_ngram_size', 3),
                **generation_args
            )
            
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

            return response
        
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            raise e
        finally:
            del inputs
            torch.cuda.empty_cache()

    async def verify(self) -> bool:
        try:
            test_question = "What is in this image?"
            test_image_path = "path/to/test/image.jpg"
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
    manager.register_adapter("phi3_vision_128k_instruct", Phi3Vision128KInstructAdapter)