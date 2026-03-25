from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

def format_example(example):

    instruction = example["query"]
    context = "\n".join(example["contexts"])
    answer = example["answer"]

    text = f"""
### Instruction:
{instruction}

### Context:
{context}

### Response:
{answer}
"""

    return {"text": text}

dataset = load_dataset("json", data_files="data/finetune_dataset.jsonl")
dataset = dataset.map(format_example)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./qwen2.5-finance-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

trainer.train()

model = model.merge_and_unload()
model.save_pretrained("./qwen2.5-finance")
tokenizer.save_pretrained("./qwen2.5-finance")