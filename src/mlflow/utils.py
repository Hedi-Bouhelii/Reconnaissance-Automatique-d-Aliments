"""
MLflow utility functions for Food Classification project
"""

import mlflow
import mlflow.pytorch
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from .config import MLFLOW_CONFIG

logger = logging.getLogger(__name__)

def setup_mlflow(tracking_uri: str = None, experiment_name: str = None):
    """Setup MLflow tracking server and experiment"""
    try:
        tracking_uri = tracking_uri or MLFLOW_CONFIG["tracking_uri"]
        experiment_name = experiment_name or MLFLOW_CONFIG["experiment_name"]
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=MLFLOW_CONFIG["artifact_location"]
            )
            logger.info(f"Created MLflow experiment: {experiment_name}")
        else:
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}")
        return False

def log_model_metrics(model, 
                      metrics: Dict[str, float], 
                      params: Dict[str, Any] = None,
                      artifact_paths: list = None):
    """Log model, metrics, and parameters to MLflow"""
    try:
        with mlflow.start_run() as run:
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Log model
            if model:
                mlflow.pytorch.log_model(model, "model")
            
            # Log artifacts
            if artifact_paths:
                for path in artifact_paths:
                    if Path(path).exists():
                        mlflow.log_artifact(path)
            
            logger.info(f"Logged to MLflow run: {run.info.run_id}")
            return run.info.run_id
            
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
        return None

def log_training_progress(epoch: int, 
                          train_loss: float, 
                          val_loss: float,
                          train_acc: float, 
                          val_acc: float,
                          learning_rate: float = None):
    """Log training progress metrics"""
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    }
    
    if learning_rate is not None:
        metrics["learning_rate"] = learning_rate
    
    mlflow.log_metrics(metrics, step=epoch)

def log_hyperparameters(params: Dict[str, Any]):
    """Log hyperparameters"""
    mlflow.log_params(params)

def log_class_performance(class_names: list, 
                         accuracies: list, 
                         f1_scores: list = None):
    """Log class-wise performance metrics"""
    for i, class_name in enumerate(class_names):
        metrics = {f"class_{class_name}_accuracy": accuracies[i]}
        
        if f1_scores and i < len(f1_scores):
            metrics[f"class_{class_name}_f1"] = f1_scores[i]
        
        mlflow.log_metrics(metrics)

def save_and_log_confusion_matrix(cm: np.ndarray, class_names: list):
    """Save and log confusion matrix"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(cm_path)
        
        # Clean up
        Path(cm_path).unlink(missing_ok=True)
        
        logger.info("Logged confusion matrix to MLflow")
        
    except Exception as e:
        logger.error(f"Failed to log confusion matrix: {e}")

def get_best_run(experiment_name: str, metric: str = "val_accuracy"):
    """Get best run from experiment based on metric"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        
        if len(runs) == 0:
            return None
        
        best_run = runs.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "metrics": best_run["metrics"],
            "params": best_run["params"],
            "tags": best_run["tags"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get best run: {e}")
        return None

def compare_models(model_names: list, stages: list = ["Staging", "Production"]):
    """Compare models in registry"""
    try:
        client = mlflow.tracking.MlflowClient()
        comparison = {}
        
        for model_name in model_names:
            model_info = {}
            
            for stage in stages:
                versions = client.get_latest_versions(model_name, stages=[stage])
                if versions:
                    version = versions[0]
                    run = client.get_run(version.run_id)
                    
                    model_info[stage] = {
                        "version": version.version,
                        "run_id": version.run_id,
                        "metrics": run.data.metrics,
                        "params": run.data.params
                    }
            
            comparison[model_name] = model_info
        
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        return None
