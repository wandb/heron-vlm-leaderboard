import os
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
import google.generativeai as genai

class GeminiAdapter(BaseAdapter):
    dependencies = [
        'pillow>=8.0.0',
        'google-generativeai',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.api_key = os.environ["GEMINI_API_KEY"]
        
        self.generation_config = {
            "temperature": self.config.get('temperature', 0.7),
            "max_output_tokens": self.config.get('max_length', 256),
        }
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name, generation_config=self.generation_config)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return 'gemini' in model_name.lower()

    async def generate_response(self, question: str, image_path: str) -> str:
        image = Image.open(image_path)
        message = [question, image]
        response = self.model.generate_content(message)

        if hasattr(response._result, 'candidates') and response._result.candidates:
            candidate = response._result.candidates[0]
            answer = "".join(part.text for part in candidate.content.parts) if candidate.content.parts else "empty response"
        else:
            answer = "Blocked by the safety filter."

        return answer

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
    manager.register_adapter("gemini", GeminiAdapter)