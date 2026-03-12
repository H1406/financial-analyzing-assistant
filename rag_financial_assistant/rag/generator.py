# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# class LocalLLMGenerator:

#     def __init__(self):

#         model_name = "microsoft/Phi-3-mini-4k-instruct"

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)

#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             dtype=torch.float16,
#             device_map="auto"
#         )

#     def generate(self, prompt):

#         inputs = self.tokenizer(prompt, return_tensors="pt").to("mps")

#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=300
#         )

#         return self.tokenizer.decode(outputs[0])

import ollama

class LocalLLMGenerator:

    def generate(self, prompt):

        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]