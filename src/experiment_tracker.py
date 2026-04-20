import mlflow
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Wraps MLflow to track every pipeline run automatically.

    What gets logged per run:
    - Parameters: method, threshold, video path, resize dims
    - Metrics: avg FPS, avg motion percentage, avg blob count
    - Tags: which mode was run
    """

    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = None
        logger.info(f"MLflow tracking — experiment: {experiment_name}")

    def start_run(self, run_name: str):
        self.run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run_name}")

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int = None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def end_run(self, summary: dict):
        """Log final summary metrics and close the run."""
        mlflow.log_metrics(summary)
        mlflow.end_run()
        logger.info(f"MLflow run ended — summary: {summary}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if mlflow.active_run():
            mlflow.end_run()
