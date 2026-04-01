"""
Fine-tuning pipeline for the Financial RAG Assistant.

Runs QLoRA (LoRA via PEFT) on top of Qwen2.5-0.5B-Instruct using SFTTrainer,
with full MLflow experiment tracking.

Usage:
    python -m rag.finetuning                      # uses config.yaml in cwd
    python -m rag.finetuning --config config.yaml
"""

import argparse
import os
import yaml
import mlflow
import mlflow.transformers

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

from mlops.tracking import (
    setup_mlflow,
    log_finetuning_params,
    log_trainable_params,
    log_train_result,
)


# ---------------------------------------------------------------------------
# Dataset formatting
# ---------------------------------------------------------------------------

def format_example(example: dict) -> dict:

    # -------- Question --------
    instruction = example["qa"]["question"]

    # -------- Context (text) --------
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))

    # -------- Context (table) --------
    table = example.get("table", [])

    table_str = ""
    if table:
        for row in table:
            table_str += " | ".join(row) + "\n"

    # -------- Combine context --------
    context = f"""
[TEXT CONTEXT]
{pre_text}

{post_text}

[TABLE CONTEXT]
{table_str}
"""

    # -------- Answer --------
    answer = example["qa"]["answer"]

    # -------- Final instruction format --------
    text = f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
{answer}
"""

    return {"text": text}


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------

class MLflowFinetuningCallback(TrainerCallback):
    """
    Logs per-step training metrics (loss, learning rate, epoch) and a final
    summary to the active MLflow run.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if not logs:
            return
        step = state.global_step
        metrics: dict[str, float] = {}
        if "loss" in logs:
            metrics["train/loss"] = logs["loss"]
        if "learning_rate" in logs:
            metrics["train/learning_rate"] = logs["learning_rate"]
        if "epoch" in logs:
            metrics["train/epoch"] = logs["epoch"]
        if "eval_loss" in logs:
            metrics["eval/loss"] = logs["eval_loss"]
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        mlflow.log_metric("total_steps", state.global_step)
        if state.best_metric is not None:
            mlflow.log_metric("best_metric", state.best_metric)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_finetuning(config_path: str = "config.yaml") -> str:
    """
    Execute the full fine-tuning pipeline and return the MLflow run ID.

    Steps:
      1. Load config, set up MLflow
      2. Log all hyperparameters
      3. Load & format dataset
      4. Load base model + apply LoRA
      5. Log trainable parameter counts
      6. Train with MLflowFinetuningCallback
      7. Log final training metrics
      8. Save LoRA adapter; log as MLflow artifact
      9. Merge adapter into base model; log merged model via mlflow.transformers
    """
    config = yaml.safe_load(open(config_path))
    ft_cfg = config["finetuning"]
    mlflow_cfg = config.get("mlflow", {})

    setup_mlflow(
        tracking_uri=mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"),
        experiment_name=mlflow_cfg.get("experiment_name", "financial_finetuning"),
    )

    with mlflow.start_run(run_name="sft_lora") as run:
        run_id = run.info.run_id
        print(f"\n[MLflow] Run ID : {run_id}")
        print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}\n")

        # --- 1. Log hyperparameters -------------------------------------------
        log_finetuning_params(ft_cfg)

        # --- 2. Dataset ----------------------------------------------------------
        dataset_path = ft_cfg["dataset_path"]
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Fine-tuning dataset not found at '{dataset_path}'.\n"
                "Generate it first by running the RAG pipeline and calling "
                "RAGPipeline.save_training_example() for each QA pair."
            )
        dataset = load_dataset("json", data_files=dataset_path)
        dataset = dataset.map(format_example)
        mlflow.log_param("dataset_size", len(dataset["train"]))
        print(f"[Dataset] {len(dataset['train'])} training examples loaded.")

        # --- 3. Model & tokenizer ------------------------------------------------
        model_name = ft_cfg["model_name"]
        print(f"[Model] Loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

        # --- 4. LoRA -------------------------------------------------------------
        lora_cfg = ft_cfg["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        log_trainable_params(model)

        # --- 5. Training arguments -----------------------------------------------
        train_cfg = ft_cfg["training"]
        training_args = TrainingArguments(
            output_dir=ft_cfg["output_dir"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            num_train_epochs=train_cfg["num_train_epochs"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            report_to="none",  # handled by MLflowFinetuningCallback
        )

        # --- 6. Train ------------------------------------------------------------
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            args=training_args,
            callbacks=[MLflowFinetuningCallback()],
        )
        print("[Training] Starting fine-tuning ...")
        train_result = trainer.train()

        # --- 7. Log final metrics ------------------------------------------------
        log_train_result(train_result)

        # --- 8. Save & log LoRA adapter weights ----------------------------------
        adapter_dir = os.path.join(ft_cfg["output_dir"], "final_adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        mlflow.log_artifacts(adapter_dir, artifact_path="lora_adapter")
        print(f"[Artifact] LoRA adapter logged to MLflow from '{adapter_dir}'.")

        # --- 9. Merge & log full model -------------------------------------------
        merged_dir = ft_cfg["merged_output_dir"]
        print("[Model] Merging LoRA weights into base model ...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        pipeline_components = {
            "model": merged_model,
            "tokenizer": tokenizer,
        }
        mlflow.transformers.log_model(
            transformers_model=pipeline_components,
            artifact_path="merged_model",
            task="text-generation",
        )
        mlflow.log_param("merged_model_local_path", os.path.abspath(merged_dir))
        print(f"[MLflow] Merged model logged. Local copy at '{merged_dir}'.")

    print(f"\nFine-tuning complete. MLflow run ID: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the financial RAG model.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()
    run_finetuning(config_path=args.config)
