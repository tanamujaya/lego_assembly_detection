"""
Configuration file for LEGO Assembly Error Detection System
Optimized for Raspberry Pi 4B deployment
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'k_folds': 1,
    'image_size': (416, 416),  # Reduced for RPi4B - can adjust to 320x320 for faster inference
    'random_seed': 42
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_CONFIG = {
    'type': 'yolov8',  # Interchangeable: yolov8, yolov5, efficientdet, etc.
    'variant': 'yolov8n',  # nano version for RPi4B (lightest)
    'pretrained': True,
    'num_classes': 2,  # Correct assembly vs Error
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_det': 100
}

# Alternative models for comparative analysis
ALTERNATIVE_MODELS = {
    'yolov8s': 'yolov8s.pt',  # Small version
    'yolov8m': 'yolov8m.pt',  # Medium version
    'yolov5n': 'yolov5n.pt',  # YOLOv5 nano
    'yolov5s': 'yolov5s.pt',  # YOLOv5 small
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    'epochs': 40,
    'batch_size': 16,  # Reduced for RPi4B memory constraints
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'patience': 20,  # Early stopping patience
    'device': 'cpu',  # RPi4B doesn't have CUDA
    'workers': 2,  # Number of data loading workers (reduced for RPi)
    'augmentation': True,
    'mixed_precision': False  # Not beneficial on CPU
}

# ============================================================================
# FEW-SHOT FINE-TUNING CONFIGURATION
# ============================================================================
FEW_SHOT_CONFIG = {
    'enabled': True,
    'num_shots': 196,  # Number of real photos per class for fine-tuning
    'fine_tune_epochs': 40,
    'fine_tune_lr': 0.0001,  # Lower learning rate for fine-tuning
    'freeze_backbone': False,  # Freeze backbone layers initially
    'unfreeze_after': 10  # Unfreeze after N epochs
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,  # HSV-Hue augmentation
    'hsv_s': 0.7,    # HSV-Saturation augmentation
    'hsv_v': 0.4,    # HSV-Value augmentation
    'degrees': 10,    # Rotation degrees
    'translate': 0.1, # Translation
    'scale': 0.5,     # Scaling
    'shear': 0.0,     # Shear
    'perspective': 0.0, # Perspective
    'flipud': 0.0,    # Vertical flip probability
    'fliplr': 0.5,    # Horizontal flip probability
    'mosaic': 1.0,    # Mosaic augmentation probability
    'mixup': 0.1      # Mixup augmentation probability
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
METRICS_CONFIG = {
    'calculate_precision': True,
    'calculate_recall': True,
    'calculate_map': True,
    'map_iou_threshold': 0.5,
    'map_iou_range': [0.5, 0.95],  # For mAP@0.5:0.95
    'save_confusion_matrix': True,
    'save_pr_curve': True
}

# ============================================================================
# RASPBERRY PI OPTIMIZATION
# ============================================================================
RPI_CONFIG = {
    'use_threading': True,
    'thread_count': 2,
    'enable_gpu': False,  # RPi4B GPU not supported by PyTorch
    'quantization': False,  # Set to True for INT8 quantization (faster inference)
    'use_onnx': False,  # Export to ONNX for optimized inference
    'use_tflite': False,  # Export to TensorFlow Lite
    'memory_efficient': True,
    'cache_predictions': False
}

# ============================================================================
# ORDER PROCESSING CONFIGURATION
# ============================================================================
ORDER_CONFIG = {
    'order_input_path': DATA_DIR / "orders",
    'reference_images_path': DATA_DIR / "reference_models",
    'real_photos_path': DATA_DIR / "captured_photos",
    'output_results_path': RESULTS_DIR / "order_results",
    'order_format': 'json',  # Format for order requests
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
INFERENCE_CONFIG = {
    'save_output_images': True,
    'output_image_path': RESULTS_DIR / "inference_outputs",
    'show_labels': True,
    'show_confidence': True,
    'color_correct': (0, 255, 0),  # Green for correct
    'color_error': (255, 0, 0),    # Red for error
    'line_thickness': 2,
    'font_scale': 0.5
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'log_file': LOGS_DIR / "system.log",
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_metrics': True,
    'metrics_file': RESULTS_DIR / "metrics.json"
}

# ============================================================================
# CLASS NAMES
# ============================================================================
CLASS_NAMES = {
    0: 'correct_assembly',
    1: 'assembly_error'
}

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================
PERFORMANCE_THRESHOLDS = {
    'min_precision': 0.85,
    'min_recall': 0.80,
    'min_map50': 0.85,
    'max_inference_time': 2.0,  # seconds per image on RPi4B
}
