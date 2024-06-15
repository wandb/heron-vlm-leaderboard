# Heron VLM Leaderboeard

## Set up
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
export OPENAI_API_KEY=<your OPENAI_API_KEY>
export HF_TOKEN=<your HF_TOKEN>
export LANG=ja_JP.UTF-8
# if needed, set the following API KEY too
export ANTHROPIC_API_KEY=<your ANTHROPIC_API_KEY>
export GEMINI_API_KEY=<your GEMINI_API_KEY>
```

## Create config.yaml file

1. create configs/config.yaml
```bash
cp configs/config_template.yaml configs/config.yaml
```
2. set each variable properly by following the below instruction

Note: This leaderboard is in its initial release and is still under development. Some features may not yet be implemented. These features will be added progressively. Thank you for your understanding and cooperation.

general:
- `wandb`: Information used for W&B support.
  - `log`: Set to True to enable logging.
  - `entity`: Name of the W&B Entity. For example, "vision-language-leaderboard".
  - `project`: Name of the W&B Project. For example, "heron-leaderboard".
  - `run_name`: Name of the W&B run. For example, "turing-motors/heron-chat-git-ja-stablelm-base-7b-v1".
  - `launch`: Set to false if you do not want to launch the run immediately.
- `testmode`: The default is false. If set to true, it allows for a lightweight implementation where only 1 or 2 questions are extracted from each category. Please set it to true when you want to perform a functionality check.
- `api`: If you don't use API, please set "api" as "false". If you use API, please select from "openai", "anthropic", "google", "cohere", "mistral", "amazon_bedrock".
- `torch_dtype`: Settings for fp16, bf16, fp32. The default is "fp16".
- `model_artifact`: Model artifact path. The default is null.
- `processor_artifact`: Processor artifact path. The default is null.
- `tokenizer_artifact`: Tokenizer artifact path. The default is null.

model:
- `_target_`: heron.models.git_llm.git_japanese_stablelm_alpha.GitJapaneseStableLMAlphaForCausalLM.from_pretrained
- `pretrained_model_name_or_path`: Name of your model. For example, "turing-motors/heron-chat-git-ja-stablelm-base-7b-v1".
- `torch_dtype`: Data type for the model. The default is "float16".
- `ignore_mismatched_sizes`: Set to true to ignore mismatched sizes. The default is true.

processor:
- `_target_`: transformers.AutoProcessor.from_pretrained
- `pretrained_model_name_or_path`: Name of the processor. For example, "turing-motors/heron-chat-git-ja-stablelm-base-7b-v1".

tokenizer:
- `_target_`: transformers.LlamaTokenizer.from_pretrained
- `pretrained_model_name_or_path`: Name of the tokenizer. For example, "novelai/nerdstash-tokenizer-v1".
- `args`:
  - `padding_side`: Padding side for the tokenizer. The default is "right".
  - `additional_special_tokens`: Additional special tokens. For example, '["▁▁"]'.

datasets:
- `llava_bench_in_the_wild_artifact_path`: Path to the LLaVA bench in the wild artifact. For example, 'vision-language-leaderboard/heron-leaderboard/llava-bench-in-the-wild:v0'.
- `llava_bench_in_the_wild_reference_path`: Path to the LLaVA bench in the wild reference. For example, 'vision-language-leaderboard/heron-leaderboard/llava-bench-in-the-wild-reference:v0'.
- `japanese_heron_bench_artifact_path`: Path to the Japanese Heron bench artifact. For example, 'vision-language-leaderboard/heron-leaderboard/japanese-heron-bench:v0'.
- `japanese_heron_bench_reference_path`: Path to the Japanese Heron bench reference. For example, 'vision-language-leaderboard/heron-leaderboard/heron-bench-reference:v0'.

generation:
- `args`:
  - `max_length`: Maximum length for generation. The default is 256.
  - `do_sample`: Set to false to disable sampling. The default is false.
  - `temperature`: Temperature for sampling. The default is 0.0.
  - `eos_token_id_list`: List of end-of-sequence token IDs. The default is '[]'.
  - `no_repeat_ngram_size`: Size of n-grams that should not be repeated. The default is 2.

   
## Evaluation execution
1. run run_eval.py
```bash
python3 run_eval.py
```
2. check the wandb dashboard
