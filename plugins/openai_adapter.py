import base64
import os
import requests
from typing import Dict, Any
from PIL import Image
from plugins.base_adapter import BaseAdapter
from tenacity import retry, stop_after_attempt, wait_fixed

class OpenAIAdapter(BaseAdapter):
    dependencies = [
        'requests>=2.25.0',
        'pillow>=8.0.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.model_name = model_name
        self.max_tokens = config.get('max_length', 4000)
        self.temperature = config.get('temperature', 0.0)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name in [
            "gpt-4-turbo", "gpt-4-turbo-2024-04-09", 
            "gpt-4o", "gpt-4o-2024-05-13",
        ]

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
    async def generate_response(self, question: str, image_path: str) -> str:
        base64_image = self.encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

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
    manager.register_adapter("openai", OpenAIAdapter)