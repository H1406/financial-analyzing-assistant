import mlflow


def start_experiment(name="rag_experiment"):
    mlflow.set_experiment(name)
    run = mlflow.start_run()
    return run


def log_config(config):
    for key, value in config.items():
        mlflow.log_param(key, value)

