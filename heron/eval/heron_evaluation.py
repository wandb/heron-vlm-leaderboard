import os
import logging
import torch
from tqdm import tqdm
import ast

import wandb
import pandas as pd
from PIL import Image

from config_singleton import WandbConfigSingleton
from common import save_results, load_questions

def process_questions(img_root, questions, verbose=True):
    instance = WandbConfigSingleton.get_instance()
    generator = instance.store['generator']
    table = wandb.Table(columns=["benchmark", "question_id", "category", "image_category", "image_file", "image", "question", "answer"])
    """
    Processes a list of questions, generating answers for each.
    """
    results = []
    for q in tqdm(questions):
        image_path = os.path.join(img_root, f"{q['image']}")
        question = q["text"]
        answer = generator.generate_response(question, image_path)
        if verbose:
            print(
                f"### ID: {q['question_id']}\n## question: {q['text']}\n## answer: {answer}\n"
            )
        q["answer"] = answer
        results.append(q)
        # add row to wandb.Table
        table.add_data("heron_bench_in_the_wild", q["question_id"], q["category"], q["image_category"], q["image"], wandb.Image(Image.open(image_path)), q["text"], q["answer"])
    return results, table

def heron_eval():
    instance = WandbConfigSingleton.get_instance()

    run = instance.run
    cfg = instance.config

    device_id = 0
    device = f"cuda:{device_id}"

    # heron-bench
    artifact = run.use_artifact(cfg.datasets.japanese_heron_bench_artifact_path, type='dataset')
    heron_dir = artifact.download()
    questions = load_questions(f"{heron_dir}/questions_ja.jsonl")
    contexts = pd.read_json(f"{heron_dir}/context_ja.jsonl", orient='records', lines=True)
    artifact = run.use_artifact(cfg.datasets.japanese_heron_bench_reference_path, type='dataset')
    ref_dir = artifact.download()
    heron_reference = load_questions(f"{ref_dir}/gpt-4-turbo-2024-04-09_answers.jsonl")

    print("Start inference")
    img_root = f"{heron_dir}/images"
    results, heron_table = process_questions(img_root, questions, verbose=True)
    print("Done inference")

    # make output dir
    output_path = './heron_output'
    os.makedirs(output_path, exist_ok=True)
    output_model_name = cfg.model.pretrained_model_name_or_path.split("/")[-1].split(".yml")[0]
    print("Saving results...")
    save_results(results, output_path, output_model_name)

    # Evaluate with GPT-4
    print("Evaluating results with GPT-4...")
    from heron_benchmark_runner import get_evaluations
    scores, judgements = get_evaluations(img_root, results, contexts, heron_reference)
    heron_table.add_column(name="judgement", data=judgements)
    heron_table.add_column(name="score", data=scores)
    #scores = evaluate_with_gpt4(results, model, processor, device)
    print("Evaluation complete")
    #print(scores)

    # table for radar chart visualization
    heron_radar_df = pd.DataFrame(data=heron_table.data, columns=heron_table.columns)
    heron_radar_df = heron_radar_df[heron_radar_df["score"] >= 1].groupby(["category"])[["score"]].mean()
    heron_radar_table = wandb.Table(dataframe=heron_radar_df.reset_index())

    # table for leaderboard
    data = heron_radar_df.mean(axis=0, numeric_only=True).to_list() + heron_radar_df.score.values.tolist()
    columns = ["ave_heron"] + ["heron_"+col for col in heron_radar_df.index.values.tolist()]
    heron_df = pd.DataFrame(data=[data], columns=columns)
    lb_df = instance.store['lb_df']
    combined_df = pd.concat([lb_df, heron_df], axis=1)
    instance.store['lb_df'] = combined_df
    combined_df.to_csv('combined_df.csv')

    run.log({"heron_table":heron_table, "heron_radar_table":heron_radar_table})