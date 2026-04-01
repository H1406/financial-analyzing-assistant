import json
import os

import yaml

from retrieval.retriever import Retriever
from rag.prompt_template import build_prompt
from rag.generator import LocalLLMGenerator

class RAGPipeline:

    def __init__(self):

        config = yaml.safe_load(open("config.yaml"))

        self.retriever = Retriever(config)

        self.generator = LocalLLMGenerator()

    def run(self, query):

        contexts = self.retriever.retrieve(query)

        prompt = build_prompt(query, contexts)

        answer = self.generator.generate(prompt)

        return answer, contexts
    @staticmethod
    def save_training_example(query: str, contexts: list, answer: str, path: str = "data/finetune_dataset.jsonl") -> None:
        """Append a QA pair to the fine-tuning dataset JSONL file."""
        example = {
            "query": query,
            "contexts": [c["text"] for c in contexts],
            "answer": answer,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(example) + "\n")
