import base64
import requests
import os
import json
import io
import logging
import torch
import ast
from PIL import Image
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from config_singleton import WandbConfigSingleton

# APIキーの取得
api_key = os.getenv('OPENAI_API_KEY')

# 画像をBase64にエンコードする関数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# for Anthropic API
# class AnthropicResponseGenerator
# To do

# for Gemini API
# class GeminiResponseGenerator
# To do

# for OpenAI API
class OpenAIResponseGenerator:
    def __init__(self, api_key, model_name="gpt-4-turbo-2024-04-09", max_tokens=4000, temperature=0.0):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
    def generate_response(self, prompt, image_path):
        """
        OpenAI APIにリクエストを送信するメソッド。
        リトライ処理を追加し、失敗した場合は例外を発生させる。
        """
        base64_image = encode_image(image_path)

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
                            "text": prompt
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

class LLMResponseGenerator:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        logging.basicConfig(level=logging.INFO)
        self.cfg = WandbConfigSingleton.get_instance().config

    def generate_response(self, question, image_path):
        image = Image.open(image_path)
        text = f"##human: {question}\n##gpt: "
        print(text)  # for debug
        inputs = self.processor(text=text, images=image, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device).half()

        logging.info(f"Input text: {text}")
        logging.info(f"Input shapes: { {k: v.shape for k, v in inputs.items()} }")
        logging.info(f"Using device: {self.device}")

        eos_token_id_list = [
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.eos_token_id,
            int(self.processor.tokenizer.convert_tokens_to_ids("\n")),
        ]
        eos_token_id_list += ast.literal_eval(self.cfg.generation.args.eos_token_id_list)

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=self.cfg.generation.args.max_length,
                    do_sample=self.cfg.generation.args.do_sample,
                    temperature=self.cfg.generation.args.temperature,
                    eos_token_id=eos_token_id_list,
                    no_repeat_ngram_size=self.cfg.generation.args.no_repeat_ngram_size,
                )
            return self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].split('##gpt:')[1]
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            logging.info(f"Inputs at error: {inputs}")
            raise e

def load_questions(path):
    """
    Loads questions from a JSONL file.
    """
    with open(path, "r") as file:
        return [json.loads(line) for line in file]

def save_results(results, output_path, model_name):
    """
    Saves the results to a JSONL file.
    """
    with open(os.path.join(output_path, f"{model_name}_answers.jsonl"), "w") as file:
        for r in results:
            file.write(json.dumps(r, ensure_ascii=False) + "\n")