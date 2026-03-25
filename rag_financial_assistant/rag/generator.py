from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
import torch

class LocalLLMGenerator:

    def __init__(self):

        model_name = "microsoft/Phi-3-mini-4k-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        gen_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=False,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=gen_config,
        )

    def generate(self, prompt):
        # no generation kwargs here
        outputs = self.pipe(prompt)
        return outputs[0]["generated_text"]
# import ollama

# class LocalLLMGenerator:

#     def generate(self, prompt):

#         response = ollama.chat(
#             model="phi3",
#             messages=[{"role": "user", "content": prompt}]
#         )

#         return response["message"]["content"]