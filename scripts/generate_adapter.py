import argparse
import os
import importlib.util
import asyncio
import ast
import traceback
import re
from typing import Dict, Any
from anthropic import Anthropic
from huggingface_hub import hf_hub_download
import timeout_decorator

class AdapterGenerator:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.max_retries = 5
        self.cache_dir = "adapter_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    async def generate_and_verify_adapter(self, model_name: str) -> str:
        error_context = ""
        last_generated_code = None
        for attempt in range(self.max_retries):
            try:
                print(f"Attempt {attempt + 1}: Generating adapter for {model_name}")
                adapter_code = self.generate_adapter(model_name, error_context)
                last_generated_code = adapter_code

                print("Preprocessing generated code")
                adapter_code = self.preprocess_code(adapter_code)

                print("Validating code syntax")
                self.validate_code_syntax(adapter_code)

                print("Verifying adapter")
                if await self.verify_adapter_code(adapter_code, model_name):
                    print(f"Adapter verified successfully for {model_name}")
                    self.save_to_cache(model_name, adapter_code)
                    return adapter_code
                else:
                    raise Exception("Adapter verification failed")

            except Exception as e:
                error_traceback = traceback.format_exc()
                error_context = f"Error during attempt {attempt + 1}:\n{error_traceback}\n\nPrevious code:\n{last_generated_code}"
                print(error_context)
                if attempt == self.max_retries - 1:
                    print(f"Failed to generate and verify adapter after {self.max_retries} attempts.")
                    raise

        return last_generated_code

    def generate_adapter(self, model_name: str, error_context: str = "") -> str:
        print(f"Downloading README for {model_name}")
        readme_path = hf_hub_download(repo_id=model_name, filename="README.md")
        with open(readme_path, "r") as f:
            readme_content = f.read()

        print("Creating adapter generation prompt")
        prompt = self._create_adapter_generation_prompt(model_name, readme_content, error_context)

        print("Generating adapter code using Anthropic API")
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        generated_code = response.content[0].text
        print("Adapter code generated successfully")
        return generated_code

    def preprocess_code(self, code: str) -> str:
        # Remove any text before the first Python code block
        code = re.sub(r'^.*?```python', '', code, flags=re.DOTALL)
        # Remove any text after the last Python code block
        code = re.sub(r'```.*$', '', code, flags=re.DOTALL)
        # Remove any remaining Markdown code block markers
        code = code.replace('```', '')
        # Strip leading and trailing whitespace
        return code.strip()

    def validate_code_syntax(self, code: str):
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Generated code contains syntax error: {str(e)}")

    @timeout_decorator.timeout(300)  # 5 minutes timeout
    async def verify_adapter_code(self, adapter_code: str, model_name: str) -> bool:
        temp_file_path = None
        try:
            print("Saving generated code to temporary file")
            temp_file_path = f"temp_{self.sanitize_filename(model_name)}_adapter.py"
            with open(temp_file_path, "w") as f:
                f.write(adapter_code)

            print("Dynamically importing generated adapter")
            spec = importlib.util.spec_from_file_location("temp_adapter", temp_file_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            print("Getting adapter class")
            adapter_class_name = self.get_adapter_class_name(model_name)
            adapter_class = getattr(temp_module, adapter_class_name)

            print("Initializing adapter")
            adapter = adapter_class(model_name, "cuda", {})

            print("Verifying adapter")
            verification_result = await adapter.verify()

            return verification_result
        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return False
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _create_adapter_generation_prompt(self, model_name: str, readme_content: str, error_context: str) -> str:
        adapter_class_name = self.get_adapter_class_name(model_name)
        base_prompt = f"""
        Given the following README.md content for the model {model_name}:

        {readme_content}

        Generate a Python code for an adapter class that can be used to run the model for the Vision and Language Leaderboard benchmark. The adapter should inherit from BaseAdapter and implement the generate_response and verify methods.

        Important: Your response should contain ONLY the Python code, without any additional text or explanations. Do not include Markdown code block markers (```python or ```) in your response.

        The adapter should follow this structure:

        import torch
        from PIL import Image
        from typing import Dict, Any
        from plugins.base_adapter import BaseAdapter
        # Add any other necessary imports, including custom model classes if needed

        class {adapter_class_name}(BaseAdapter):
            dependencies = [
                'torch>=1.9.0',
                'transformers>=4.20.0',
                'pillow>=8.0.0',
                # Add any other specific dependencies required by this model
            ]

            def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
                super().__init__(model_name, device)
                self.config = config
                # Initialize the model and any other necessary components

            @classmethod
            def supports_model(cls, model_name: str) -> bool:
                return model_name.startswith("{model_name}")

            async def generate_response(self, question: str, image_path: str) -> str:
                # Implement the generate_response method

            async def verify(self) -> bool:
                # Implement the verify method
                # Use "test.jpg" for this purpose as it is included in this repository

            @classmethod
            def get_config_schema(cls) -> Dict[str, Any]:
                # Define the configuration parameters specific to this model

        def register_plugin(manager):
            manager.register_adapter("{model_name.split('/')[-1].lower()}", {adapter_class_name})

        Ensure that the generated code is syntactically correct and can be parsed without errors. Implement proper error handling and logging in all methods.
        """

        if error_context:
            base_prompt += f"\n\nThe previous attempt to generate this adapter encountered the following error:\n{error_context}\nPlease adjust the code to address this issue and ensure that the adapter works correctly."

        return base_prompt

    def get_adapter_class_name(self, model_name: str) -> str:
        class_name = re.sub(r'[^a-zA-Z0-9]', ' ', model_name.split('/')[-1]).title().replace(' ', '')
        return f"{class_name}Adapter"

    def sanitize_filename(self, filename: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)

    def save_to_cache(self, model_name: str, adapter_code: str):
        cache_file = os.path.join(self.cache_dir, f"{self.sanitize_filename(model_name)}_adapter.py")
        with open(cache_file, "w") as f:
            f.write(adapter_code)

async def main():
    parser = argparse.ArgumentParser(description="Generate an adapter for a new model")
    parser.add_argument("model_name", type=str, help="The name or path of the model")
    args = parser.parse_args()

    generator = AdapterGenerator()
    try:
        adapter_code = await generator.generate_and_verify_adapter(args.model_name)
        adapter_file_path = f"plugins/{generator.sanitize_filename(args.model_name)}_adapter.py"
        with open(adapter_file_path, "w") as f:
            f.write(adapter_code)
        print(f"Adapter generated for {args.model_name}")
        print(f"Adapter code saved to {adapter_file_path}")
    except Exception as e:
        print(f"Failed to generate and verify adapter: {str(e)}")
        error_file_path = f"error_{generator.sanitize_filename(args.model_name)}.log"
        with open(error_file_path, "w") as f:
            f.write(traceback.format_exc())
        print(f"Error details saved to {error_file_path}")

if __name__ == "__main__":
    asyncio.run(main())