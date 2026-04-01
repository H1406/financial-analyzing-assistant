import mlflow
import mlflow.pytorch
from typing import Any


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking URI and set the active experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _flatten_config(cfg: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict for MLflow param logging."""
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_config(value, prefix=full_key))
        elif isinstance(value, list):
            result[full_key] = str(value)
        else:
            result[full_key] = value
    return result


def log_finetuning_params(ft_config: dict) -> None:
    """Log all finetuning hyperparameters (flattened) to the active MLflow run."""
    flat = _flatten_config(ft_config)
    mlflow.log_params(flat)


def log_trainable_params(model) -> None:
    """Log trainable vs. total parameter counts and percentage."""
    trainable, total = model.get_nb_trainable_parameters()
    mlflow.log_params({
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / total, 4),
    })


def log_train_result(train_result) -> None:
    """Log final training metrics from a Transformers TrainOutput."""
    metrics = {
        "final/train_loss": train_result.training_loss,
        "final/train_runtime_sec": train_result.metrics.get("train_runtime", 0),
        "final/samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "final/steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
    }
    mlflow.log_metrics(metrics)


# ---------------------------------------------------------------------------
# Backward-compatible helpers (used by RAG pipeline)
# ---------------------------------------------------------------------------

def start_experiment(name: str = "rag_experiment"):
    mlflow.set_experiment(name)
    return mlflow.start_run()


def log_config(config: dict) -> None:
    for key, value in config.items():
        mlflow.log_param(key, value)
