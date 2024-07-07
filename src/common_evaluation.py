import os
import json
import logging
from typing import Dict, Any, List, Tuple
import wandb
import pandas as pd
from PIL import Image
from tqdm import tqdm
from omegaconf import ListConfig, DictConfig, OmegaConf
from plugins.base_adapter import BaseAdapter
from src.llm_judge import LLMJudge

async def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as file:
        return [json.loads(line) for line in file]

def save_results(results: List[Dict[str, Any]], output_path: str, output_model_name: str) -> None:
    output_file = os.path.join(output_path, f"{output_model_name}_result.jsonl")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
async def process_questions(
    adapter: BaseAdapter,
    img_root: str,
    questions: List[Dict[str, Any]],
    benchmark_config: Dict[str, Any],
    verbose: bool = True
) -> Tuple[List[Dict[str, Any]], wandb.Table]:
    table_columns = benchmark_config.get('table_columns', [])
    if isinstance(table_columns, ListConfig):
        table_columns = list(table_columns)
    
    table = wandb.Table(columns=table_columns)
    
    results = []
    for q in tqdm(questions):
        image_key = benchmark_config.get('image_key')
        question_key = benchmark_config.get('question_key')
        
        if not image_key or not question_key:
            raise ValueError("image_key and question_key must be specified in the benchmark config")
        
        image_path = os.path.join(img_root, f"{q[image_key]}")
        question = q[question_key]
        answer = await adapter.generate_response(question, image_path)
        
        if verbose:
            print(f"### ID: {q.get('question_id', 'N/A')}\n## question: {question}\n## answer: {answer}\n")
        
        q["answer"] = answer
        results.append(q)
        
        table_data = []
        for column in table_columns:
            if column == 'benchmark':
                table_data.append(benchmark_config['name'])
            elif column == 'image':
                table_data.append(wandb.Image(Image.open(image_path)))
            elif column == 'question':
                table_data.append(question)
            elif column == 'answer':
                table_data.append(answer)
            elif column in q:
                table_data.append(q[column])
            else:
                table_data.append(None)
        
        table.add_data(*table_data)
    
    return results, table

async def evaluate_benchmark(adapter: BaseAdapter, benchmark_name: str) -> None:
    run = wandb.run
    config_dict = dict(wandb.config)
    cfg = OmegaConf.create(config_dict)
    benchmark_config = cfg.benchmarks[benchmark_name]

    # Download and prepare data
    artifact = run.use_artifact(benchmark_config['artifact_path'], type='dataset')
    data_dir = artifact.download()
    questions = await load_questions(f"{data_dir}/{benchmark_config['questions_file']}")
    contexts = pd.read_json(f"{data_dir}/{benchmark_config['context_file']}", orient='records', lines=True)

    # Download reference answers
    ref_artifact = run.use_artifact(benchmark_config['reference_path'], type='dataset')
    ref_dir = ref_artifact.download()
    references = await load_questions(f"{ref_dir}/{benchmark_config['reference_file']}")

    # Process questions
    img_root = f"{data_dir}/images"
    results, table = await process_questions(adapter, img_root, questions, benchmark_config, verbose=True)

    # Save results
    output_path = f'./{benchmark_config["name"]}_output'
    os.makedirs(output_path, exist_ok=True)
    output_model_name = cfg.model.pretrained_model_name_or_path.split("/")[-1].split(".yml")[0]
    save_results(results, output_path, output_model_name)

    # Evaluate with LLM
    llm_judge = LLMJudge(img_root)
    scores, judgements = await llm_judge.evaluate_responses(results, references, contexts, benchmark_config)

    # Convert tuples to lists
    judgements = list(judgements)
    scores = list(scores)

    table.add_column(name="judgement", data=judgements)
    table.add_column(name="score", data=scores)

    # Create radar chart data
    radar_df = pd.DataFrame(data=table.data, columns=table.columns)
    radar_df = radar_df[radar_df["score"] >= 1].groupby(["category"])[["score"]].mean()
    radar_table = wandb.Table(dataframe=radar_df.reset_index())

    # Update leaderboard data
    data = radar_df.mean(axis=0, numeric_only=True).to_list() + radar_df.score.values.tolist()
    columns = [f"ave_{benchmark_config['name']}"] + [f"{benchmark_config['name']}_{col}" for col in radar_df.index.values.tolist()]
    benchmark_df = pd.DataFrame(data=[data], columns=columns)
    
    lb_dict = run.summary.get('lb_dict', {"model_name":cfg.model.pretrained_model_name_or_path})
    lb_df = pd.DataFrame(columns=lb_dict.keys(), data=[lb_dict.values()])
    combined_df = pd.concat([lb_df, benchmark_df], axis=1)
    run.summary['lb_dict'] = combined_df.iloc[0].to_dict()

    # Log results
    run.log({f"{benchmark_config['name']}_table": table, 
             f"{benchmark_config['name']}_radar_table": radar_table})