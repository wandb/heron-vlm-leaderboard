import io
import base64
import os
import time
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from anthropic import Anthropic, InternalServerError

class ClaudeAdapter(BaseAdapter):
    dependencies = [
        'pillow>=8.0.0',
        'anthropic',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.client = Anthropic(api_key=self.api_key)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith('claude')

    async def generate_response(self, question: str, image_path: str) -> str:
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Encode the image to base64
                image_data = self.encode_image_to_base64(image_path)
                
                # Prepare the messages for the API request
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                        ],
                    },
                ]

                # Make the API request
                response = self.client.messages.create(
                    max_tokens=self.config.get('max_length', 1000),
                    messages=messages,
                    temperature=self.config.get('temperature', 0.7),
                    model=self.model_name,
                )

                # Return the generated response
                return response.content[0].text

            except InternalServerError as e:
                if attempt < max_retries - 1:
                    print(f"Internal server error occurred. Retrying in {retry_delay} seconds. (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached. Last error: {str(e)}")
                    raise e

        raise Exception("Maximum number of retries exceeded.")

    def encode_image_to_base64(self, filepath, max_size=5*1024*1024*3//4):
        with Image.open(filepath) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=85)
            size = buffer.tell()

            if size > max_size:
                quality = 85
                while size > max_size and quality > 10:
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", optimize=True, quality=quality)
                    size = buffer.tell()
                    quality -= 5
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

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
            "api_key": str,
            "max_length": int,
            "temperature": float,
        }

def register_plugin(manager):
    manager.register_adapter("claude", ClaudeAdapter)