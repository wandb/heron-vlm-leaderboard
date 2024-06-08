import os
import json

def load_questions(path):
    """
    Loads questions from a JSONL file.
    """
    with open(path, "r") as file:
        return [json.loads(line) for line in file]

def save_results(results, output_path, model_name):
    file_path = os.path.join(output_path, f"{model_name}_answers.jsonl")
    with open(file_path, 'w', encoding='utf-8') as file:
        for r in results:
            file.write(json.dumps(r, ensure_ascii=False) + "\n")