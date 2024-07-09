# Heron Vision Language Leaderboard

## Overview

This project is a benchmarking tool for evaluating and comparing the performance of various Vision Language Models (VLMs). It uses two datasets: LLaVA-Bench-In-the-Wild and Japanese HERON Bench to measure model performance.

## Key Features

- Benchmark evaluation for multiple VLMs
- Logging and visualization using Weights & Biases
- Flexible configuration options (using Hydra configuration system)

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

## Usage

1. Generate model adapter (if needed):
   If the adapter for your chosen model doesn't exist in the `plugins` directory, you can either use the `generate_adapter.py` script to automatically create one, or implement it yourself by referring to existing adapters:
   ```
   python scripts/generate_adapter.py <model_name>
   ```
   Replace `<model_name>` with the name or path of your model.

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
