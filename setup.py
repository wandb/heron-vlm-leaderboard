from setuptools import setup, find_packages

setup(
    name='heron_vlm_leaderboard',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'pillow>=8.0.0',
        'hydra-core',
        'omegaconf',
        'wandb',
        'pandas',
        'tqdm',
        'openai',
        'anthropic',
        'tenacity',
        'huggingface_hub',
        'tiktoken',
        'matplotlib',
        'torchvision',
        'transformers_stream_generator',
    ],
    entry_points={
        'console_scripts': [
            'run_eval=run_eval:main',
        ],
    },
)