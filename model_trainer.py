"""
Model Training Module for LEGO Assembly Error Detection
Supports multiple model architectures with K-fold cross-validation and few-shot fine-tuning
"""

import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available. Install with: pip install ultralytics")


class ModelTrainer:
    """Handles model training with support for multiple architectures"""
    
    def __init__(self, config):
        self.config = config
        self.model_config = config.MODEL_CONFIG
        self.training_config = config.TRAINING_CONFIG
        self.few_shot_config = config.FEW_SHOT_CONFIG
        self.models_dir = config.MODELS_DIR
        self.results_dir = config.RESULTS_DIR
        
        # Create model-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"training_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def train_yolo_model(self, 
                        data_yaml: str,
                        model_name: Optional[str] = None,
                        save_name: str = "best_model.pt") -> Dict:
        """
        Train YOLO model (v5 or v8)
        
        Args:
            data_yaml: Path to dataset YAML configuration
            model_name: Model variant name (default from config)
            save_name: Name for saved model
            
        Returns:
            Dictionary with training results
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")
        
        if model_name is None:
            model_name = self.model_config['variant']
        
        logger.info(f"Starting YOLO training with {model_name}")
        logger.info(f"Dataset: {data_yaml}")
        
        # Initialize model
        if self.model_config['pretrained']:
            model = YOLO(f"{model_name}.pt")
            logger.info(f"Loaded pretrained {model_name}")
        else:
            model = YOLO(f"{model_name}.yaml")
            logger.info(f"Initialized {model_name} from scratch")
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': self.training_config['epochs'],
            'batch': self.training_config['batch_size'],
            'imgsz': self.config.DATASET_CONFIG['image_size'][0],
            'device': self.training_config['device'],
            'workers': self.training_config['workers'],
            'patience': self.training_config['patience'],
            'save': True,
            'project': str(self.run_dir),
            'name': 'train',
            'exist_ok': True,
            'pretrained': self.model_config['pretrained'],
            'optimizer': self.training_config['optimizer'],
            'lr0': self.training_config['learning_rate'],
            'verbose': True,
            'plots': True,
        }
        
        # Add augmentation if enabled
        if self.training_config['augmentation']:
            aug_config = self.config.AUGMENTATION_CONFIG
            train_args.update({
                'hsv_h': aug_config['hsv_h'],
                'hsv_s': aug_config['hsv_s'],
                'hsv_v': aug_config['hsv_v'],
                'degrees': aug_config['degrees'],
                'translate': aug_config['translate'],
                'scale': aug_config['scale'],
                'shear': aug_config['shear'],
                'perspective': aug_config['perspective'],
                'flipud': aug_config['flipud'],
                'fliplr': aug_config['fliplr'],
                'mosaic': aug_config['mosaic'],
                'mixup': aug_config['mixup'],
            })
        
        # Train model
        start_time = time.time()
        results = model.train(**train_args)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save best model
        best_model_path = self.models_dir / save_name
        model.save(str(best_model_path))
        logger.info(f"Best model saved to {best_model_path}")
        
        # Collect results
        training_results = {
            'model_type': model_name,
            'training_time': training_time,
            'best_model_path': str(best_model_path),
            'training_dir': str(self.run_dir / 'train'),
            'epochs_completed': len(results.results_dict) if hasattr(results, 'results_dict') else self.training_config['epochs'],
            'config': train_args
        }
        
        # Save training info
        with open(self.run_dir / 'training_info.json', 'w') as f:
            json.dump(training_results, f, indent=2)
        
        return training_results
    
    def train_with_kfold(self, fold_info_path: str) -> List[Dict]:
        """
        Train model with K-fold cross-validation
        
        Args:
            fold_info_path: Path to fold information JSON file
            
        Returns:
            List of results for each fold
        """
        # Load fold information
        with open(fold_info_path, 'r') as f:
            fold_info = json.load(f)
        
        logger.info(f"Starting K-fold cross-validation with {len(fold_info)} folds")
        
        all_results = []
        
        for fold in fold_info:
            fold_num = fold['fold']
            yaml_path = fold['yaml_path']
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Training Fold {fold_num}/{len(fold_info)}")
            logger.info(f"{'='*50}")
            
            # Train model for this fold
            save_name = f"fold_{fold_num}_best.pt"
            fold_results = self.train_yolo_model(
                data_yaml=yaml_path,
                save_name=save_name
            )
            
            fold_results['fold'] = fold_num
            fold_results.update(fold)
            all_results.append(fold_results)
            
            logger.info(f"Fold {fold_num} completed")
        
        # Calculate average metrics across folds
        avg_results = self._calculate_kfold_averages(all_results)
        
        # Save K-fold results
        kfold_results = {
            'individual_folds': all_results,
            'average_results': avg_results,
            'num_folds': len(fold_info)
        }
        
        results_path = self.results_dir / 'kfold_results.json'
        with open(results_path, 'w') as f:
            json.dump(kfold_results, f, indent=2)
        
        logger.info(f"\nK-fold cross-validation completed")
        logger.info(f"Results saved to {results_path}")
        
        return all_results
    
    def fine_tune_few_shot(self,
                          base_model_path: str,
                          few_shot_yaml: str,
                          save_name: str = "few_shot_model.pt") -> Dict:
        """
        Fine-tune model with few-shot learning
        
        Args:
            base_model_path: Path to base trained model
            few_shot_yaml: Path to few-shot dataset YAML
            save_name: Name for fine-tuned model
            
        Returns:
            Dictionary with fine-tuning results
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not installed")
        
        logger.info(f"Starting few-shot fine-tuning")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Few-shot data: {few_shot_yaml}")
        
        # Load base model
        model = YOLO(base_model_path)
        
        # Freeze backbone if configured
        if self.few_shot_config['freeze_backbone']:
            logger.info("Freezing backbone layers")
            self._freeze_backbone(model)
        
        # Fine-tuning arguments
        finetune_args = {
            'data': few_shot_yaml,
            'epochs': self.few_shot_config['fine_tune_epochs'],
            'batch': self.training_config['batch_size'],
            'imgsz': self.config.DATASET_CONFIG['image_size'][0],
            'device': self.training_config['device'],
            'workers': self.training_config['workers'],
            'patience': self.training_config['patience'],
            'save': True,
            'project': str(self.run_dir),
            'name': 'finetune',
            'exist_ok': True,
            'optimizer': self.training_config['optimizer'],
            'lr0': self.few_shot_config['fine_tune_lr'],
            'verbose': True,
            'plots': True,
        }
        
        # Train with frozen backbone
        start_time = time.time()
        results = model.train(**finetune_args)
        
        # Unfreeze and continue training if configured
        if self.few_shot_config['freeze_backbone'] and \
           self.few_shot_config['unfreeze_after'] > 0:
            logger.info(f"Unfreezing backbone after {self.few_shot_config['unfreeze_after']} epochs")
            self._unfreeze_backbone(model)
            
            # Continue training
            remaining_epochs = self.few_shot_config['fine_tune_epochs'] - \
                             self.few_shot_config['unfreeze_after']
            if remaining_epochs > 0:
                finetune_args['epochs'] = remaining_epochs
                results = model.train(**finetune_args, resume=True)
        
        finetuning_time = time.time() - start_time
        
        logger.info(f"Fine-tuning completed in {finetuning_time:.2f} seconds")
        
        # Save fine-tuned model
        finetuned_model_path = self.models_dir / save_name
        model.save(str(finetuned_model_path))
        logger.info(f"Fine-tuned model saved to {finetuned_model_path}")
        
        # Collect results
        finetune_results = {
            'base_model': base_model_path,
            'finetuned_model_path': str(finetuned_model_path),
            'finetuning_time': finetuning_time,
            'training_dir': str(self.run_dir / 'finetune'),
            'config': finetune_args
        }
        
        # Save fine-tuning info
        with open(self.run_dir / 'finetuning_info.json', 'w') as f:
            json.dump(finetune_results, f, indent=2)
        
        return finetune_results
    
    def train_alternative_model(self, 
                               model_type: str,
                               data_yaml: str,
                               save_name: str = None) -> Dict:
        """
        Train alternative model architecture for comparison
        
        Args:
            model_type: Type of model (from ALTERNATIVE_MODELS in config)
            data_yaml: Path to dataset YAML
            save_name: Name for saved model
            
        Returns:
            Dictionary with training results
        """
        if model_type not in self.config.ALTERNATIVE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if save_name is None:
            save_name = f"{model_type}_best.pt"
        
        logger.info(f"Training alternative model: {model_type}")
        
        # Use the same training function but with different model
        results = self.train_yolo_model(
            data_yaml=data_yaml,
            model_name=model_type,
            save_name=save_name
        )
        
        results['model_type'] = model_type
        
        return results
    
    def _freeze_backbone(self, model):
        """Freeze backbone layers of the model"""
        try:
            # For YOLO models, freeze all layers except the head
            for param in model.model.parameters():
                param.requires_grad = False
            
            # Unfreeze head (detection layers)
            if hasattr(model.model, 'model'):
                # YOLOv8 structure
                for param in model.model.model[-1].parameters():
                    param.requires_grad = True
            
            logger.info("Backbone frozen successfully")
        except Exception as e:
            logger.warning(f"Could not freeze backbone: {e}")
    
    def _unfreeze_backbone(self, model):
        """Unfreeze all model layers"""
        try:
            for param in model.model.parameters():
                param.requires_grad = True
            logger.info("All layers unfrozen")
        except Exception as e:
            logger.warning(f"Could not unfreeze backbone: {e}")
    
    def _calculate_kfold_averages(self, fold_results: List[Dict]) -> Dict:
        """Calculate average metrics across K-folds"""
        avg_results = {
            'avg_training_time': np.mean([f['training_time'] for f in fold_results]),
            'std_training_time': np.std([f['training_time'] for f in fold_results]),
            'total_training_time': np.sum([f['training_time'] for f in fold_results])
        }
        
        return avg_results
    
    def export_model(self, 
                    model_path: str, 
                    format: str = 'onnx',
                    optimize: bool = True) -> str:
        """
        Export model to different formats for deployment
        
        Args:
            model_path: Path to trained model
            format: Export format (onnx, tflite, etc.)
            optimize: Whether to optimize for Raspberry Pi
            
        Returns:
            Path to exported model
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not installed")
        
        logger.info(f"Exporting model to {format} format")
        
        model = YOLO(model_path)
        
        # Export arguments
        export_args = {
            'format': format,
            'imgsz': self.config.DATASET_CONFIG['image_size'][0],
            'optimize': optimize,
            'half': False,  # No FP16 on CPU
        }
        
        if format == 'onnx':
            export_args['dynamic'] = False
            export_args['simplify'] = True
        
        # Export model
        export_path = model.export(**export_args)
        
        logger.info(f"Model exported to {export_path}")
        
        return str(export_path)
