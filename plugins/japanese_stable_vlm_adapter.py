import torch
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoModelForVision2Seq, AutoImageProcessor, AutoTokenizer

class JapaneseStableVLMAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if config.get('torch_dtype') == "fp16":
            model_kwargs["variant"] = "fp16"
            model_kwargs["torch_dtype"] = torch.float16
        elif config.get('torch_dtype') == "int8":
            model_kwargs["variant"] = "fp16"
            model_kwargs["load_in_8bit"] = True
            model_kwargs["max_memory"] = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

        self.model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith('stabilityai/japanese-stable-vlm')

    async def generate_response(self, question: str, image_path: str) -> str:
        image = Image.open(image_path)
        prompt = self.build_prompt(task="vqa", input=question)
        
        inputs = self.processor(images=[image], return_tensors="pt")
        text_encoding = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.update(text_encoding)
        
        generation_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.config.get('max_length', 256),
            "temperature": self.config.get('temperature', 0.7),
            "min_length": 1,
            "top_p": 0,
            "no_repeat_ngram_size": self.config.get('no_repeat_ngram_size', 3),
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
            print(f"Error during model generation: {e}")
            raise e
        finally:
            del inputs
            del outputs
            torch.cuda.empty_cache()

    def build_prompt(self, task="vqa", input=None, sep="\n\n### "):
        TASK2INSTRUCTION = {
            "caption": "画像を詳細に述べてください。",
            "tag": "与えられた単語を使って、画像を詳細に述べてください。",
            "vqa": "与えられた画像を下に、質問に答えてください。",
        }
        assert task in TASK2INSTRUCTION, f"Please choose from {list(TASK2INSTRUCTION.keys())}"
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

    async def verify(self) -> bool:
        try:
            test_question = "この画像には何が写っていますか？"
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
            "no_repeat_ngram_size": int,
            "torch_dtype": str,
        }

def register_plugin(manager):
    manager.register_adapter("japanese_stable_vlm", JapaneseStableVLMAdapter)