"""
Multi-View Configuration for LEGO Assembly Error Detection
Extends the base configuration to support 4-view inspection system
"""

# ============================================================================
# MULTI-VIEW CONFIGURATION
# ============================================================================
MULTIVIEW_CONFIG = {
    'enabled': True,
    'num_views': 4,
    'view_angles': [0, 90, 180, 270],  # Degrees
    
    # View naming convention in your dataset
    # Pattern: image_NNNN_AAA.png where NNNN=sequential number, AAA=angle
    'view_suffix_format': '_{angle}',  # e.g., image_0000_0.png, image_0000_90.png
    
    # Dataset structure patterns
    'pattern_1': {
        'range': (0, 3199),  # image_0000 to image_3199
        'images_per_view': 10,  # 10 consecutive images = same view
        'description': 'First 3200 images: 10 images per view, cycling through 0°, 90°, 180°, 270°'
    },
    'pattern_2': {
        'range': (3200, 6399),  # image_3200 to image_6399
        'images_per_view': 2,  # 2 consecutive images = same view
        'description': 'Last 3200 images: 2 images per view, cycling through 0°, 90°, 180°, 270°'
    },
    
    # Multi-view decision logic
    'decision_strategy': 'any_error',  # Options: 'any_error', 'majority_vote', 'all_error'
    # 'any_error': If ANY view shows error → Model = INCORRECT (recommended for safety)
    # 'majority_vote': If majority of views show error → Model = INCORRECT
    # 'all_error': If ALL views show error → Model = INCORRECT
    
    # Confidence aggregation for multi-view
    'confidence_aggregation': 'max',  # Options: 'max', 'mean', 'min'
    # 'max': Use highest confidence across views (recommended)
    # 'mean': Average confidence across views
    # 'min': Use lowest confidence (most conservative)
    
    # View-specific weights (if using weighted voting)
    'view_weights': {
        0: 1.0,    # Front view (0°)
        90: 1.0,   # Right view (90°)
        180: 1.0,  # Back view (180°)
        270: 1.0   # Left view (270°)
    },
    
    # Deployment settings
    'capture_all_views': True,  # Capture all 4 views before making decision
    'turntable_rotation_delay': 0.5,  # Seconds to wait after rotation
    'camera_stabilization_delay': 0.3,  # Seconds to wait before capture
    
    # Performance monitoring
    'track_per_view_performance': True,  # Track accuracy per view angle
    'save_per_view_results': True,  # Save individual view predictions
}

# ============================================================================
# TRAINING STRATEGIES FOR MULTI-VIEW
# ============================================================================
MULTIVIEW_TRAINING_STRATEGIES = {
    # Strategy 1: Train on all individual views (recommended)
    'individual_views': {
        'description': 'Train on each view independently, aggregate during deployment',
        'train_on': 'all_views',  # Each view is a separate training sample
        'aggregate_during': 'inference',  # Combine predictions at deployment
        'benefits': [
            'Maximum training data utilization',
            'Model learns view-specific features',
            'Simple training pipeline'
        ]
    },
    
    # Strategy 2: Train on view groups
    'view_groups': {
        'description': 'Group multiple views of same model as multi-label sample',
        'train_on': 'grouped_views',
        'aggregate_during': 'training',
        'benefits': [
            'Model learns view relationships',
            'Better for models with temporal/spatial understanding'
        ]
    },
    
    # Current recommendation: individual_views strategy
    'selected_strategy': 'individual_views'
}

# ============================================================================
# MODEL GROUPING (for deployment)
# ============================================================================
def get_model_id_from_filename(filename: str) -> int:
    """
    Extract model ID from filename
    
    For your naming scheme (image_NNNN_AAA.png):
    - Pattern 1 (0000-3199): Every 40 images = 1 model (10 imgs × 4 views)
    - Pattern 2 (3200-6399): Every 8 images = 1 model (2 imgs × 4 views)
    
    Args:
        filename: Image filename (e.g., 'image_0000_0.png')
    
    Returns:
        Model ID number
    """
    # Extract sequential number from filename
    # Assumes format: image_NNNN_AAA.png
    base_name = filename.split('_')
    if len(base_name) < 2:
        raise ValueError(f"Invalid filename format: {filename}")
    
    seq_num = int(base_name[1])
    
    # Determine model ID based on pattern
    if seq_num < 3200:
        # Pattern 1: 10 images per view, 40 images per model
        model_id = seq_num // 40
    else:
        # Pattern 2: 2 images per view, 8 images per model
        # Add 800 (number of models in pattern 1: 3200/40 = 80)
        adjusted_num = seq_num - 3200
        model_id = 800 + (adjusted_num // 8)
    
    return model_id


def get_view_angle_from_filename(filename: str) -> int:
    """
    Extract view angle from filename
    
    Args:
        filename: Image filename (e.g., 'image_0000_90.png')
    
    Returns:
        View angle (0, 90, 180, or 270)
    """
    # Extract angle from filename
    # Assumes format: image_NNNN_AAA.png where AAA is angle
    parts = filename.replace('.png', '').replace('.jpg', '').split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")
    
    angle = int(parts[2])
    if angle not in [0, 90, 180, 270]:
        raise ValueError(f"Invalid angle {angle} in filename: {filename}")
    
    return angle


# ============================================================================
# EVALUATION METRICS FOR MULTI-VIEW
# ============================================================================
MULTIVIEW_METRICS_CONFIG = {
    # Standard metrics (calculated per individual view)
    'per_view_metrics': [
        'precision',
        'recall',
        'f1_score',
        'mAP@0.5',
        'mAP@0.5:0.95'
    ],
    
    # Multi-view aggregated metrics (calculated per model)
    'per_model_metrics': [
        'model_accuracy',  # Correct decision on entire model (all 4 views)
        'view_agreement',  # How often all views agree
        'error_detection_rate',  # Successfully detected error in at least 1 view
        'false_positive_rate',  # Incorrectly flagged correct model as error
    ],
    
    # View-specific analysis
    'view_analysis': {
        'best_performing_view': True,  # Which view angle is most accurate
        'worst_performing_view': True,  # Which view angle has most errors
        'view_complementarity': True,  # How well views complement each other
        'critical_view_identification': True  # Which views catch errors others miss
    },
    
    # Confusion matrix for multi-view
    'multiview_confusion_matrix': True,  # Track: (predicted, actual) at model level
}

# ============================================================================
# DEPLOYMENT CONFIGURATION FOR MULTI-VIEW
# ============================================================================
MULTIVIEW_DEPLOYMENT_CONFIG = {
    # Hardware setup
    'turntable': {
        'enabled': True,
        'type': 'stepper_motor',  # or 'servo', 'manual'
        'steps_per_revolution': 200,  # For stepper motor
        'microstepping': 16,  # Microstepping ratio
        'rotation_speed': 60,  # RPM
        'gpio_pins': {
            'step': 17,
            'direction': 27,
            'enable': 22
        }
    },
    
    # Camera setup
    'camera': {
        'type': 'pi_camera',  # or 'usb_camera'
        'resolution': (320, 320),  # Match training resolution
        'warm_up_frames': 10,  # Discard first N frames
        'capture_format': 'png'
    },
    
    # Inspection workflow
    'workflow': {
        'home_position': 0,  # Starting angle
        'rotation_sequence': [0, 90, 180, 270],  # Order of capture
        'return_to_home': True,  # Rotate back to 0° after inspection
        'batch_processing': False  # Process all views at once vs one-by-one
    },
    
    # Result visualization
    'visualization': {
        'save_annotated_images': True,
        'show_per_view_results': True,
        'create_summary_image': True,  # 4-panel view with results
        'highlight_error_views': True
    }
}

# ============================================================================
# DATA AUGMENTATION FOR MULTI-VIEW
# ============================================================================
MULTIVIEW_AUGMENTATION_CONFIG = {
    # View-consistent augmentation (same for all views of a model)
    'consistent_augmentation': {
        'enabled': True,
        'augmentations': [
            'brightness',
            'contrast',
            'saturation',
            'gaussian_noise'
        ]
    },
    
    # View-independent augmentation (different for each view)
    'independent_augmentation': {
        'enabled': True,
        'augmentations': [
            'rotation_small',  # Small rotations (±5°)
            'translation',
            'scale'
        ]
    },
    
    # Cross-view augmentation (simulate different lighting per view)
    'cross_view_augmentation': {
        'enabled': True,
        'simulate_lighting_changes': True,  # Different lighting per view
        'shadow_variation': True
    }
}

# ============================================================================
# SUMMARY
# ============================================================================
"""
QUICK START GUIDE FOR MULTI-VIEW TRAINING:

1. Your dataset structure (already done):
   - image_0000_0.png, image_0000_90.png, image_0000_180.png, image_0000_270.png
   - image_0001_0.png, image_0001_90.png, ...
   
2. Training approach (recommended):
   - Train YOLO on ALL individual views as separate samples
   - Each view is treated as independent training example
   - Model learns: "Is THIS view correct or incorrect?"
   
3. Deployment approach:
   - Capture 4 views (0°, 90°, 180°, 270°)
   - Run inference on each view
   - Aggregate results using 'any_error' strategy:
     * If ANY view predicts error → Final decision = INCORRECT
     * Only if ALL views predict correct → Final decision = CORRECT
   
4. Expected dataset:
   - Pattern 1: 3200 images → 800 models (10 images per view × 4 views = 40 imgs/model)
   - Pattern 2: 3200 images → 400 models (2 images per view × 4 views = 8 imgs/model)
   - Total: 6400 images → 1200 models
   
5. Training command (using your existing system):
   ```bash
   python main.py train \
     --mode kfold \
     --dataset ./data/renamed_renders \
     --model yolov8n \
     --k 10 \
     --epochs 100
   ```
   
   The existing training pipeline will automatically handle all views!
"""
