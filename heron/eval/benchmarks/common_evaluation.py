import os
import logging
import wandb
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config_singleton import WandbConfigSingleton
from common import save_results, load_questions
from benchmarks.benchmark_runner import get_evaluations

def process_questions(img_root, questions, benchmark_config, verbose=True):
    instance = WandbConfigSingleton.get_instance()
    generator = instance.store['generator']
    table = wandb.Table(columns=benchmark_config['table_columns'])
    
    results = []
    for q in tqdm(questions):
        image_path = os.path.join(img_root, f"{q[benchmark_config['image_key']]}")  # 更新
        question = q[benchmark_config['question_key']]
        answer = generator.generate_response(question, image_path)
        if verbose:
            print(f"### ID: {q['question_id']}\n## question: {question}\n## answer: {answer}\n")
        q["answer"] = answer
        results.append(q)
        
        table_data = [benchmark_config['name'], q["question_id"], q["category"], q[benchmark_config['image_key']],  # 更新
                      wandb.Image(Image.open(image_path)), question, answer]
        if "image_category" in q and "image_category" in benchmark_config['table_columns']:
            table_data.insert(benchmark_config['table_columns'].index("image_category"), q["image_category"])
        table.add_data(*table_data)
    
    return results, table

def evaluate_benchmark(benchmark_config):
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    # Download and prepare data
    artifact = run.use_artifact(benchmark_config['artifact_path'], type='dataset')
    data_dir = artifact.download()
    questions = load_questions(f"{data_dir}/{benchmark_config['questions_file']}")
    contexts = pd.read_json(f"{data_dir}/{benchmark_config['context_file']}", orient='records', lines=True)

    # Download reference answers
    ref_artifact = run.use_artifact(benchmark_config['reference_path'], type='dataset')
    ref_dir = ref_artifact.download()
    references = load_questions(f"{ref_dir}/{benchmark_config['reference_file']}")

    # Process questions
    img_root = f"{data_dir}/images"
    results, table = process_questions(img_root, questions, benchmark_config, verbose=True)

    # Save results
    output_path = f'./{benchmark_config["name"]}_output'
    os.makedirs(output_path, exist_ok=True)
    output_model_name = cfg.model.pretrained_model_name_or_path.split("/")[-1].split(".yml")[0]
    save_results(results, output_path, output_model_name)

    # Evaluate with GPT-4
    scores, judgements = get_evaluations(img_root, results, contexts, references, benchmark_config)
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
    
    lb_df = instance.store['lb_df']
    combined_df = pd.concat([lb_df, benchmark_df], axis=1)
    instance.store['lb_df'] = combined_df
    combined_df.to_csv(benchmark_config['leaderboard_csv'])

    # Log results
    run.log({f"{benchmark_config['name']}_table": table, 
             f"{benchmark_config['name']}_radar_table": radar_table})