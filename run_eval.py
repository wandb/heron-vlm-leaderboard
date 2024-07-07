import asyncio
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import pandas as pd
from src.plugin_manager import PluginManager
from src.common_evaluation import evaluate_benchmark
from src.caching import disk_cache

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))

async def async_main(cfg: DictConfig) -> None:
    # Convert Hydra configuration to dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print("Debug: Full config")
    print(OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=config_dict,
        name=cfg.wandb.run_name
    )
    config_dict = dict(wandb.config)
    # Convert Python dictionary to DictConfig
    cfg = OmegaConf.create(config_dict)

    plugin_manager = PluginManager("plugins")
    adapter = plugin_manager.get_adapter(
        cfg.model.pretrained_model_name_or_path,
        f"cuda:{cfg.device_id}",
        cfg.generation.args,
    )

    # Iterate through each benchmark defined in the configuration
    for benchmark_name, benchmark_config in cfg.benchmarks.items():
        print(f"Debug: Processing benchmark: {benchmark_name}")
        print(f"Debug: Benchmark config: {OmegaConf.to_yaml(benchmark_config)}")
        results = await evaluate_benchmark(adapter, benchmark_name)
        wandb.log({f"{benchmark_name}_results": results})

    # Log final results
    lb_dict = wandb.run.summary.get('lb_dict', {})
    lb_df = pd.DataFrame(columns=lb_dict.keys(), data=[lb_dict.values()])
    if not lb_df.empty:
        # Create a list of benchmark names
        benchmark_names = cfg.benchmarks.keys()
        
        # Create a dataframe for radar chart
        radar_df = lb_df.drop(['model_name'] + [f"ave_{name}" for name in benchmark_names], axis=1)
        radar_df = radar_df.T.reset_index()
        radar_df.columns = ['category', 'score']
        
        # Log to wandb
        wandb.log({
            "lb_table": wandb.Table(dataframe=lb_df),
            "radar_table": wandb.Table(dataframe=radar_df)
        })
    else:
        print("Warning: lb_df is empty. Cannot log results.")

    wandb.finish()

if __name__ == "__main__":
    main()