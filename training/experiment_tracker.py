import mlflow
import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, tracking_dir="mlflow_tracking"):
        self.tracking_dir = tracking_dir
        if not os.path.exists(self.tracking_dir):
            os.makedirs(self.tracking_dir)
            
    def log_experiment_json(self, model_name, params, metrics):
        """
        Logs experiment to a local JSON file if MLflow server is not configured.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.tracking_dir, f"experiment_{model_name}_{timestamp}.json")
        
        record = {
            "model_name": model_name,
            "timestamp": timestamp,
            "params": params,
            "metrics": metrics
        }
        
        with open(log_file, "w") as f:
            json.dump(record, f, indent=4)
            
        print(f"[Tracker] Logged experiment for {model_name} -> {log_file}")
