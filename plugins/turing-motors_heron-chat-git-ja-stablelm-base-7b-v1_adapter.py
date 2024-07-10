import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoProcessor, LlamaTokenizer
from heron.models.git_llm.git_japanese_stablelm_alpha import GitJapaneseStableLMAlphaForCausalLM
import logging

class HeronChatGitJaStablelmBase7BV1Adapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
        'heron',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.model = GitJapaneseStableLMAlphaForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, ignore_mismatched_sizes=True
            )
            self.model.eval()
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "novelai/nerdstash-tokenizer-v1",
                padding_side="right",
                additional_special_tokens=["▁▁"],
            )
            self.processor.tokenizer = self.tokenizer
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("turing-motors/heron-chat-git-ja-stablelm-base-7b-v1")

    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            text = f"##human: {question}\n##gpt: "

            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                truncation=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=256, do_sample=False, temperature=0., no_repeat_ngram_size=2)

            response = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            return response.split("##gpt: ")[-1].strip()
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Error generating response"

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
            "max_length": {
                "type": "integer",
                "default": 256,
                "description": "Maximum length of generated text"
            },
            "temperature": {
                "type": "number",
                "default": 0.0,
                "description": "Sampling temperature"
            },
            "no_repeat_ngram_size": {
                "type": "integer",
                "default": 2,
                "description": "Size of n-grams to prevent repetition"
            }
        }

def register_plugin(manager):
    manager.register_adapter("heron-chat-git-ja-stablelm-base-7b-v1", HeronChatGitJaStablelmBase7BV1Adapter)