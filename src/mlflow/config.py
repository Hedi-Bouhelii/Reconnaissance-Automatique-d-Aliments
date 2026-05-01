"""
MLflow configuration settings for Food Classification project
"""

import os
from pathlib import Path

# MLflow Configuration
MLFLOW_CONFIG = {
    # Tracking server settings
    "tracking_uri": "http://localhost:5000",
    "experiment_name": "food-classification",
    "registry_uri": "http://localhost:5000",
    
    # Model registry settings
    "model_registry": {
        "name": "food-classification-models",
        "staging_alias": "staging",
        "production_alias": "production"
    },
    
    # Artifact settings
    "artifact_location": str(Path("./mlruns")),
    
    # Default run settings
    "default_run": {
        "tags": {
            "project": "food-classification",
            "version": "1.0",
            "framework": "pytorch"
        }
    }
}

# Dataset configuration
DATASET_CONFIG = {
    "name": "food-101",
    "version": "1.0",
    "num_classes": 101,
    "selected_classes": 10,
    "image_size": (224, 224),
    "batch_size": 32,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
}

# Model configuration
MODEL_CONFIG = {
    "resnet50": {
        "name": "resnet50",
        "pretrained": True,
        "num_classes": 101,
        "input_size": (224, 224)
    },
    "resnet101": {
        "name": "resnet101", 
        "pretrained": True,
        "num_classes": 101,
        "input_size": (224, 224)
    },
    "efficientnet_b4": {
        "name": "efficientnet_b4",
        "pretrained": True,
        "num_classes": 101,
        "input_size": (380, 380)
    }
}

# Training configuration
TRAINING_CONFIG = {
    "max_epochs": 50,
    "early_stopping_patience": 10,
    "learning_rates": [1e-4, 1e-3, 1e-2, 1e-5],
    "optimizers": ["adam", "sgd", "adamw"],
    "batch_sizes": [16, 32, 64],
    "weight_decay": [1e-4, 1e-5, 1e-3]
}
