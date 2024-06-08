import base64
import requests
import os
import logging
import torch
import ast
import hydra
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed
from config_singleton import WandbConfigSingleton

from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
from heron.models.git_llm.git_japanese_stablelm_alpha import GitJapaneseStableLMAlphaForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer, AutoProcessor

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor
from datasets import load_dataset

import requests
from PIL import Image

HERON_TYPE1_LIST = [
    "turing-motors/heron-chat-git-ja-stablelm-base-7b-v1",
    "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1-llava-620k",
    "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1",
    "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0",
]
JAPANESE_STABLEVLM_LIST = [
    'stabilityai/japanese-stable-vlm',
]
QWEN_VL_LIST = [
    'Qwen/Qwen-VL-Chat',
]

def load_processor(cfg):
    if cfg.tokenizer is None:
        processor = hydra.utils.instantiate(cfg.processor, _recursive_=False)
    else:
        tokenizer_args = {}
        if cfg.tokenizer.args is not None:
            tokenizer_args = {k: v for k, v in cfg.tokenizer.args.items() if v is not None}
        if tokenizer_args.get("additional_special_tokens"):
            additional_special_tokens = ast.literal_eval(tokenizer_args['additional_special_tokens'])
            del tokenizer_args['additional_special_tokens']
            tokenizer = hydra.utils.call(cfg.tokenizer, **tokenizer_args, _recursive_=False)
            tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        else:
            tokenizer = hydra.utils.call(cfg.tokenizer, **tokenizer_args, _recursive_=False)
        processor = hydra.utils.call(cfg.processor, _recursive_=False)
        processor.tokenizer = tokenizer
    return processor

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

class HeronType1ResponseGenerator:
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

# for japanese-stable-vlm
# helper function to format input prompts
TASK2INSTRUCTION = {
    "caption": "画像を詳細に述べてください。",
    "tag": "与えられた単語を使って、画像を詳細に述べてください。",
    "vqa": "与えられた画像を下に、質問に答えてください。",
}

def build_prompt(task="vqa", input=None, sep="\n\n### "):
    assert (
        task in TASK2INSTRUCTION
    ), f"Please choose from {list(TASK2INSTRUCTION.keys())}"
    if task in ["tag", "vqa"]:
        assert input is not None, "Please fill in `input`!"
        if task == "tag" and isinstance(input, list):
            input = "、".join(input)
    else:
        assert input is None, f"`{task}` mode doesn't support to input questions"
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    instruction = TASK2INSTRUCTION[task]
    msgs = [": \n" + instruction, ": \n"]
    if input:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + input)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

class JapanseseStableVLMResponseGenerator:
    def __init__(self, model, processor, tokenizer, device):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = WandbConfigSingleton.get_instance().config

    @torch.inference_mode()
    def generate_response(self, question, image_path):
        image = Image.open(image_path)
        prompt = build_prompt(task="vqa", input=question)
        
        inputs = self.processor(images=[image], return_tensors="pt")
        text_encoding = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.update(text_encoding)
        
        generation_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.cfg.generation.args.max_length,
            "temperature":self.cfg.generation.args.temperature,
            "min_length": 1,
            "top_p": 0,
            "no_repeat_ngram_size": self.cfg.generation.args.no_repeat_ngram_size,
        }
        
        try:
            outputs = self.model.generate(
                **inputs.to(self.device, dtype=self.model.dtype), 
                **generation_kwargs
            )
            generated = [
                txt.strip() for txt in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ]
            return generated[0]
        
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            raise e
        finally:
            del inputs
            del outputs
            torch.cuda.empty_cache()


# for Qwen/Qwen-VL-Chat
class QwenVLChatResponseGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = WandbConfigSingleton.get_instance().config

    @torch.inference_mode()
    def generate_response(self, question, image_path):
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': question},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response


def get_adapter():
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    if cfg.api:
        if cfg.api=="openai":
            generator = OpenAIResponseGenerator(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name=cfg.model.pretrained_model_name_or_path,
                max_tokens=cfg.generation.args.max_length,
                temperature=cfg.generation.args.temperature,
            )
            instance = WandbConfigSingleton.get_instance()
            instance.store['generator'] = generator
            return generator
        
    elif cfg.model.pretrained_model_name_or_path in HERON_TYPE1_LIST:
        device_id = 0
        device = f"cuda:{device_id}"

        # Model settings
        if cfg.torch_dtype == "bf16":
            torch_dtype: torch.dtype = torch.bfloat16
        elif cfg.torch_dtype == "fp16":
            torch_dtype = torch.float16
        elif cfg.torch_dtype == "fp32":
            torch_dtype = torch.float32
        else:
            raise ValueError("torch_dtype must be bf16 or fp16. Other types are not supported.")
        model = hydra.utils.call(cfg.model, torch_dtype=torch_dtype, _recursive_=False)
        model = model.half()
        model.eval()
        model.to(device)
        print("Model loaded")

        # Processor settings
        processor = load_processor(cfg)
        print("Processor loaded")
        generator = HeronType1ResponseGenerator(model, processor, device)

        return generator

    elif cfg.model.pretrained_model_name_or_path in JAPANESE_STABLEVLM_LIST:
        device_id = 0
        device = f"cuda:{device_id}"

        max_length = cfg.generation.args.max_length
        model_path = cfg.model.pretrained_model_name_or_path
        model_name = model_path

        load_in = cfg.torch_dtype # @param ["fp32", "fp16", "int8"]
        # @markdown If you use Colab free plan, please set `load_in` to `int8`. But, please remember that `int8` degrades the performance. In general, `fp32` is better than `fp16` and `fp16` is better than `int8`.

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if load_in == "fp16":
            model_kwargs["variant"] = "fp16"
            model_kwargs["torch_dtype"] = torch.float16
        elif load_in == "int8":
            model_kwargs["variant"] = "fp16"
            model_kwargs["load_in_8bit"] = True
            model_kwargs["max_memory"] = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

        model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
        processor = AutoImageProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model.eval()
        model.to(device)
        print('Load model')

        generator = JapanseseStableVLMResponseGenerator(model, processor, tokenizer, device)

        return generator

    elif cfg.model.pretrained_model_name_or_path in QWEN_VL_LIST:
        device_id = 0
        device = f"cuda:{device_id}"

        max_length = cfg.generation.args.max_length
        model_path = cfg.model.pretrained_model_name_or_path
        model_name = model_path

        #load_in = cfg.torch_dtype # @param ["fp32", "fp16", "int8"]
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}

        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        generator = QwenVLChatResponseGenerator(model, tokenizer, device)

        return generator
