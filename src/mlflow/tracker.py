"""
MLflow tracker for Food Classification project
"""

import mlflow
import mlflow.pytorch
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging
from datetime import datetime

from .config import MLFLOW_CONFIG, DATASET_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class MLflowTracker:
    """MLflow tracking wrapper for Food Classification experiments"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or MLFLOW_CONFIG["experiment_name"]
        self.tracking_uri = MLFLOW_CONFIG["tracking_uri"]
        self.current_run = None
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking server and experiment"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_registry_uri(MLFLOW_CONFIG["registry_uri"])
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=MLFLOW_CONFIG["artifact_location"]
                )
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
                
            self.experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new MLflow run"""
        try:
            # Combine default tags with provided tags
            combined_tags = MLFLOW_CONFIG["default_run"]["tags"].copy()
            if tags:
                combined_tags.update(tags)
            
            self.current_run = mlflow.start_run(
                run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                experiment_id=self.experiment_id,
                tags=combined_tags
            )
            
            # Log dataset configuration
            self.log_params(DATASET_CONFIG, prefix="dataset")
            
            logger.info(f"Started MLflow run: {self.current_run.info.run_id}")
            return self.current_run
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def end_run(self):
        """End the current MLflow run"""
        if self.current_run:
            mlflow.end_run()
            self.current_run = None
            logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters to MLflow"""
        try:
            formatted_params = {}
            for key, value in params.items():
                param_name = f"{prefix}.{key}" if prefix else key
                formatted_params[param_name] = value
            
            mlflow.log_params(formatted_params)
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model(self, model, model_name: str, artifact_path: str = "model"):
        """Log PyTorch model to MLflow"""
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=model_name
            )
            logger.info(f"Logged model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_artifact(self, file_path: str, artifact_path: str = None):
        """Log artifact to MLflow"""
        try:
            mlflow.log_artifact(file_path, artifact_path)
            logger.info(f"Logged artifact: {file_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_training_metrics(self, 
                           train_loss: float,
                           val_loss: float,
                           train_acc: float,
                           val_acc: float,
                           epoch: int,
                           learning_rate: float = None):
        """Log training metrics"""
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "epoch": epoch
        }
        
        if learning_rate:
            metrics["learning_rate"] = learning_rate
        
        self.log_metrics(metrics, step=epoch)
    
    def log_class_metrics(self, 
                         class_names: List[str],
                         class_accuracies: List[float],
                         class_f1_scores: List[float] = None):
        """Log class-wise metrics"""
        for i, class_name in enumerate(class_names):
            metrics = {
                f"class_{class_name}_accuracy": class_accuracies[i]
            }
            
            if class_f1_scores and i < len(class_f1_scores):
                metrics[f"class_{class_name}_f1"] = class_f1_scores[i]
            
            self.log_metrics(metrics)
    
    def log_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Log confusion matrix as artifact"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save as artifact
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            
            self.log_artifact(cm_path)
            Path(cm_path).unlink()  # Clean up
            
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
    
    def log_hyperparameter_search_results(self, 
                                        study_results: Dict[str, Any],
                                        best_params: Dict[str, Any],
                                        best_value: float):
        """Log hyperparameter search results"""
        # Log study summary
        self.log_params({
            "n_trials": study_results.get("n_trials", 0),
            "best_value": best_value,
            "study_time": study_results.get("study_time", 0)
        }, prefix="hyperparameter_search")
        
        # Log best parameters
        self.log_params(best_params, prefix="best_params")
    
    def register_model(self, 
                      model_name: str, 
                      run_id: str = None,
                      stage: str = "Staging"):
        """Register model in MLflow Model Registry"""
        try:
            model_uri = f"runs:/{run_id or self.current_run.info.run_id}/model"
            
            mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            # Transition to specified stage
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(model_name)[-1].version
            
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage
            )
            
            logger.info(f"Registered model {model_name} v{model_version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
    
    def get_model_info(self, model_name: str, stage: str = "Staging"):
        """Get model information from registry"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            model_version = client.get_latest_versions(model_name, stages=[stage])
            if not model_version:
                return None
                
            version_info = model_version[0]
            run_id = version_info.run_id
            
            # Get run details
            run = client.get_run(run_id)
            
            return {
                "name": model_name,
                "version": version_info.version,
                "stage": version_info.current_stage,
                "run_id": run_id,
                "creation_timestamp": version_info.creation_timestamp,
                "last_updated_timestamp": version_info.last_updated_timestamp,
                "run_params": run.data.params,
                "run_metrics": run.data.metrics,
                "tags": run.data.tags
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    def load_model_from_registry(self, model_name: str, stage: str = "Staging"):
        """Load model from MLflow Model Registry"""
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pytorch.load_model(model_uri)
            
            logger.info(f"Loaded model {model_name} from {stage}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
