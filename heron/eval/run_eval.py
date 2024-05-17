import sys
sys.path.append(".")
import ast
import hydra
from omegaconf import DictConfig, OmegaConf
from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
from heron.models.git_llm.git_japanese_stablelm_alpha import GitJapaneseStableLMAlphaForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer, AutoProcessor
import torch
import wandb
import pandas as pd
from config_singleton import WandbConfigSingleton
#from cleanup import cleanup_gpu

import json
import os
import requests
from PIL import Image

import fire

import yaml
from tqdm import tqdm

from heron.models.prepare_processors import get_processor
from heron.models.utils import load_model, load_pretrained_weight

from common import load_questions, LLMResponseGenerator, OpenAIResponseGenerator
from llava_evaluation import llava_bench_itw_eval
from heron_evaluation import heron_eval

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


@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Show configurations

    # WandB settings
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=cfg_dict,
        job_type="evaluation",
    )
    if cfg.wandb.launch:
        cfg_dict = run.config.as_dict()  # for launch
        assert isinstance(cfg_dict, dict)
        cfg = OmegaConf.create(cfg_dict)
        assert isinstance(cfg, DictConfig)

    # Initialize the WandbConfigSingleton
    wandb_store = {
        "generator":None, 
        "lb_df":pd.DataFrame({"model_name": [cfg.model.pretrained_model_name_or_path]})}
    WandbConfigSingleton.initialize(run, wandb_store)
    cfg = WandbConfigSingleton.get_instance().config

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
    else:
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
        generator = LLMResponseGenerator(model, processor, device)

        instance = WandbConfigSingleton.get_instance()
        instance.store['generator'] = generator
    
    # llava-bench
    llava_bench_itw_eval()

    # heron-bench
    heron_eval()

    # log results
    instance = WandbConfigSingleton.get_instance()
    lb_df = instance.store['lb_df']
    radar_df = lb_df.drop(['model_name', 'ave_llava_itw', 'ave_heron'], axis=1).T.reset_index()
    radar_df.columns = ['category', 'score']
    run.log({"lb_table": wandb.Table(dataframe=lb_df), "radar_table": wandb.Table(dataframe=radar_df)})

    # finish run
    run.finish()

if __name__ == "__main__":
    main()