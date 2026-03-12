from retrieval.retriever import Retriever
from rag.prompt_template import build_prompt
from rag.generator import LocalLLMGenerator
import yaml
import json

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
    def save_training_example(query, contexts, answer):

        example = {
            "query": query,
            "contexts": [c["text"] for c in contexts],
            "answer": answer
        }

        with open("data/finetune_dataset.jsonl", "a") as f:
            f.write(json.dumps(example) + "\n")
