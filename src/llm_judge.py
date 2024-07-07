import asyncio
import os
import base64
from typing import Dict, Any, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIError
import pandas as pd

class LLMJudge:
    def __init__(self, img_root):
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.img_root = img_root

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(APIError)
    )
    async def evaluate_response(
        self,
        question: Dict[str, Any],
        reference: Dict[str, Any],
        contexts: pd.DataFrame,
        benchmark_config: Dict[str, Any]
    ) -> Tuple[int, str]:

        context = contexts.loc[contexts["image"] == question['image'], benchmark_config['context_key']].values[0]
        prompt = self._create_evaluation_prompt(question, reference, context, benchmark_config)
        
        image_path = os.path.join(self.img_root, f"{question[benchmark_config['image_key']]}")
        base64_image = self.encode_image(image_path)

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates AI-generated responses."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000,
            temperature=0.5,
        )

        evaluation = response.choices[0].message.content
        score = self._extract_score(evaluation)
        return score, evaluation

    def _create_evaluation_prompt(self, question, reference, context, benchmark_config):
        return f"""You are a helpful assistant.
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by comparing the assistant's answer with the reference answer. Be as objective as possible. The expected language is Japanese. Responses in languages other than Japanese will incur score deductions unless specifically required. Failure to use Japanese at all will result in the lowest evaluation. However, using Japanese is not mandatory when providing only Python scripts or calculation results, where Japanese is not essential. Additionally, your explanation of judgement should be in Japanese. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
        [Context]
        {context}

        [Question]
        {question[benchmark_config['question_key']]}
        
        [The Start of Reference Answer]
        {reference['answer']}
        [The End of Reference Answer]

        [The Start of Assistant's Answer]
        {question['answer']}
        [The End of Assistant's Answer]
        """

    def _extract_score(self, evaluation: str) -> int:
        import re
        match = re.search(r'\[\[(\d+)\]\]', evaluation)
        return int(match.group(1)) if match else -1

    async def evaluate_responses(
        self,
        questions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        contexts: pd.DataFrame,
        benchmark_config: Dict[str, Any]
    ) -> Tuple[List[int], List[str]]:
        tasks = [
            self.evaluate_response(q, r, contexts, benchmark_config)
            for q, r in zip(questions, references)
        ]
        results = await asyncio.gather(*tasks)
        return list(zip(*results))