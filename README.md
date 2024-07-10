![Header Image](header.png)

# [Heron Vision Language Leaderboard](http://vlm.nejumi.ai)

## Overview

This project is a benchmarking tool for evaluating and comparing the performance of various Vision Language Models (VLMs). It uses two datasets: LLaVA-Bench-In-the-Wild and Japanese HERON Bench to measure model performance.

## Key Features

- Benchmark evaluation for multiple VLMs
- Logging and visualization using Weights & Biases
- Flexible configuration options (using Hydra configuration system)
- Experimental adapter generator for new models

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/wandb/heron-vlm-leaderboard.git
   cd heron-vlm-leaderboard
   ```

2. Install the project:
   ```
   pip install -e .
   pip install -r requirements.txt
   ```
   This will install the project and its dependencies listed in `setup.py` and `requirements.txt`.

3. Install additional dependencies for your chosen model:
   Depending on the model you want to use, you may need to install additional libraries. Refer to the model's documentation for specific requirements.

## API Key Setup

Before using the benchmarking tool and adapter generator, you need to set up the following API keys:

1. **WANDB_API_KEY**: Required for logging and visualization with Weights & Biases.
2. **OPENAI_API_KEY**: Required for some model evaluations.
3. **ANTHROPIC_API_KEY**: Required for the experimental adapter generator.

Set these environment variables before running the scripts:

```bash
export WANDB_API_KEY=your_wandb_api_key
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

1. Generate model adapter (if needed):
   If the adapter for your chosen model doesn't exist in the `plugins` directory, you can use the experimental `generate_adapter.py` script to automatically create one:
   ```
   python scripts/generate_adapter.py <model_name>
   ```
   Replace `<model_name>` with the name or path of your Hugging Face model.

   **How it works**:
   - The script downloads the README.md from the Hugging Face model card.
   - It uses Claude-3.5-Sonnet (via the Anthropic API) to generate adapter code based on the model's documentation.
   - If errors occur during generation or verification, the script will retry up to 5 times, incorporating error feedback into each subsequent attempt.

   **Note**: The adapter generator is an experimental feature and may require manual adjustments to the generated code. It may not work for all models, especially those with unique architectures or requirements. Always review and test the generated adapter before using it in evaluation.

   **Troubleshooting**:
   - If the generator fails, check the error logs in the `error_<model_name>.log` file.
   - You may need to manually adjust the generated code based on the specific requirements of your model.

   Alternatively, you can implement the adapter yourself by referring to existing adapters in the `plugins` directory.

2. Configuration:
   Customize benchmark settings in the `config.yaml` file. You can also add new benchmarks by adding configuration files to the `configs/benchmarks` directory. This allows for easy expansion of the evaluation suite with custom benchmarks.

3. Run the evaluation:
   ```
   python3 run_eval.py
   ```

## Benchmark Datasets

The datasets (LLaVA-Bench-In-the-Wild and Japanese HERON Bench) will be automatically downloaded using Weights & Biases Artifacts when you run the evaluation.

1. LLaVA-Bench-In-the-Wild (Japanese version)
2. Japanese HERON Bench

## Result Visualization

Use Weights & Biases to track and visualize benchmark results in real-time.

## License

This project is released under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

We would like to express our gratitude to Turing Inc. for their technical support in Vision and Language models and evaluation methodologies. Their expertise and contributions have been invaluable to this project.
