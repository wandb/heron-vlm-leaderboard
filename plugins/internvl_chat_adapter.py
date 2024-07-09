import torch
from PIL import Image
from typing import Dict, Any, List
from plugins.base_adapter import BaseAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torchvision import transforms as T
import logging

class InternVLChatAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
        'torchvision',
        'sentencepiece',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.device = device
        self.use_flash_attention = config.get('use_flash_attention', False)
        self.use_single_gpu = config.get('use_single_gpu', False)
        self.device_map = 'auto' if not self.use_single_gpu else None

        # Flash Attention configuration
        if self.use_flash_attention:
            try:
                from transformers.utils import is_flash_attn_available
                if not is_flash_attn_available():
                    print("Flash Attention is not available. Falling back to default attention.")
                    self.use_flash_attention = False
            except ImportError:
                print("Flash Attention import failed. Falling back to default attention.")
                self.use_flash_attention = False

        # Get and update model configuration
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.vision_config.use_flash_attn = self.use_flash_attention
        if hasattr(config, 'llm_config'):
            config.llm_config.attn_implementation = "flash_attention_2" if self.use_flash_attention else "eager"

        # Initialize the model
        model_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }

        if not self.use_single_gpu:
            model_kwargs["device_map"] = self.device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if self.use_single_gpu:
            self.model = self.model.eval().to(self.device)
        else:
            self.model = self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("OpenGVLab/InternVL-Chat")

    def load_image(self, image_path: str, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def build_transform(self, input_size: int):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def dynamic_preprocess(self, image: Image.Image, min_num: int = 1, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = False) -> List[Image.Image]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images

    def find_closest_aspect_ratio(self, aspect_ratio: float, target_ratios: List[tuple], width: int, height: int, image_size: int) -> tuple:
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @torch.inference_mode()
    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            pixel_values = self.load_image(image_path, max_num=6).to(torch.bfloat16)
            if not self.device_map:
                pixel_values = pixel_values.to(self.device)

            generation_config = {
                "num_beams": 1,
                "max_new_tokens": self.config.get('max_length', 512),
                "do_sample": self.config.get('do_sample', True),
                "temperature": self.config.get('temperature', 0.7),
                "no_repeat_ngram_size": self.config.get('no_repeat_ngram_size', 3),
            }

            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            return response

        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            raise e
        finally:
            if not self.device_map:
                torch.cuda.empty_cache()

    async def verify(self) -> bool:
        try:
            test_question = "What can you see in this image?"
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
            "use_flash_attention": bool,
            "device_map": (str, type(None)),
        }

def register_plugin(manager):
    manager.register_adapter("internvl_chat", InternVLChatAdapter)