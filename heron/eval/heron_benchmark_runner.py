import base64
import requests
import os
import re
import json
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from vandl_adapter import encode_image, OpenAIResponseGenerator

def load_questions(path):
    """
    Loads questions from a JSONL file.
    """
    with open(path, "r") as file:
        return [json.loads(line) for line in file]
        
# To Do
from concurrent.futures import ThreadPoolExecutor

def get_evaluations(img_root, results, contexts, references, verbose=True):
    def process_question(index, q, r):
        image_path = os.path.join(img_root, f"{q['image']}")
        question = q["text"]
        answer = q["answer"]
        reference = r["answer"]
        context = contexts.loc[contexts["image"]==q['image'], 'caption'].values[0]

        prompt = f"""You are a helpful assistant.
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by comparing the assistant's answer with the reference answer. Be as objective as possible. The expected language is Japanese. Responses in languages other than Japanese will incur score deductions unless specifically required. Failure to use Japanese at all will result in the lowest evaluation. However, using Japanese is not mandatory when providing only Python scripts or calculation results, where Japanese is not essential. Additionally, your explanation of judgement should be in Japanese. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".
        [Context]
        {context}

        [Question]
        {question}
        
        [The Start of Reference Answer]
        {reference}
        [The End of Reference Answer]

        [The Start of Assistant's Answer]
        {answer}
        [The End of Assistant's Answer]
        """
        generator = OpenAIResponseGenerator(
            api_key=os.getenv('OPENAI_API_KEY'),
        )

        result = generator.generate_response(prompt, image_path)
        print(result)
        match = re.search(r": \[\[(\d+)\]\]", result)
        if match:
            score = int(match.group(1))
        else:
            score = -1
        return index, score, result, prompt, answer, q

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_question, i, q, r) for i, (q, r) in enumerate(zip(results, references))]
        
        scores = [None] * len(results)
        judgements = [None] * len(results)
        prompts = [None] * len(results)
        answers = [None] * len(results)
        for future in futures:
            index, score, result, prompt, answer, q = future.result()
            scores[index] = score
            judgements[index] = result
            prompts[index] = prompt
            answers[index] = answer
            if verbose:
                print(
                    f"### ID: {q['question_id']}\n## prompt: {prompts[index]}\n## evaluation: {result}\n"
                )
            q["answer"] = answers[index]
            results[index] = q

    return scores, judgements