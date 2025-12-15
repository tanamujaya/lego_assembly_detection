"""
Training Pipeline for LEGO Assembly Error Detection
Supports K-fold cross-validation and standard train/val/test split
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from config import Config
from data_preparation import DataPreparation
from model_manager import ModelManager


class TrainingPipeline:
    """Training pipeline with K-fold cross-validation"""
    
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.config.setup_directories()
        self.data_prep = DataPreparation(self.config)
        self.model_manager = ModelManager(self.config)
        
    def train_single_fold(self, model_type: str, data_yaml: Path, 
                         fold_num: int = None, **kwargs) -> Dict:
        """
        Train a single fold or standard split
        
        Args:
            model_type: Type of model to train
            data_yaml: Path to dataset YAML file
            fold_num: Fold number (None for standard split)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        if fold_num is not None:
            print(f"Training Fold {fold_num} with {model_type}")
        else:
            print(f"Training {model_type} on standard split")
        print(f"{'='*60}\n")
        
        # Get model instance
        model = self.model_manager.get_model(model_type)
        model.load_model()
        
        # Train
        train_name = f"{model_type}_fold{fold_num}" if fold_num else f"{model_type}_standard"
        train_results = model.train(data_yaml, name=train_name, **kwargs)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_metrics = model.evaluate(data_yaml, split='val')
        
        # Evaluate on test set if available
        test_metrics = {}
        if fold_num is None:  # Only evaluate test for standard split
            print("\nEvaluating on test set...")
            test_metrics = model.evaluate(data_yaml, split='test')
        
        # Combine results
        results = {
            'model_type': model_type,
            'fold': fold_num,
            'timestamp': datetime.now().isoformat(),
            'train_results': train_results,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        return results
    
    def train_kfold(self, model_type: str, dataset_dir: Path, 
                   k: int = None, save_all_models: bool = False) -> Dict:
        """
        Train with K-fold cross-validation
        
        Args:
            model_type: Type of model to train
            dataset_dir: Directory containing the dataset
            k: Number of folds (defaults to config.K_FOLDS)
            save_all_models: Whether to save all fold models or just best
            
        Returns:
            Dictionary with aggregated results across all folds
        """
        if k is None:
            k = self.config.K_FOLDS
        
        print(f"\n{'='*60}")
        print(f"Starting {k}-Fold Cross-Validation with {model_type}")
        print(f"{'='*60}\n")
        
        # Organize dataset
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        dataset = self.data_prep.organize_dataset(images_dir, labels_dir)
        
        if len(dataset) == 0:
            raise ValueError(f"No data found in {dataset_dir}")
        
        print(f"Total samples: {len(dataset)}")
        
        # Create K-fold splits
        folds = self.data_prep.create_kfold_splits(dataset, k=k)
        
        # Prepare separate test set (15% of total data)
        from sklearn.model_selection import train_test_split
        kfold_data, test_data = train_test_split(
            dataset, 
            test_size=self.config.TEST_SPLIT,
            random_state=42
        )
        
        # Re-create folds with remaining data
        folds = self.data_prep.create_kfold_splits(kfold_data, k=k)
        
        # Prepare YOLO datasets for each fold
        output_dir = self.config.DATA_DIR / 'kfold_datasets'
        yaml_paths = self.data_prep.prepare_kfold_datasets(
            folds, output_dir, f"{model_type}_kfold"
        )
        
        # Train each fold
        fold_results = []
        all_val_metrics = {
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50-95': []
        }
        
        start_time = time.time()
        
        for fold_num, yaml_path in enumerate(yaml_paths, 1):
            fold_result = self.train_single_fold(
                model_type, 
                yaml_path, 
                fold_num=fold_num
            )
            
            fold_results.append(fold_result)
            
            # Collect metrics
            val_metrics = fold_result['val_metrics']
            for metric in all_val_metrics.keys():
                if metric in val_metrics:
                    all_val_metrics[metric].append(val_metrics[metric])
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        for metric, values in all_val_metrics.items():
            if values:
                aggregate_metrics[f'{metric}_mean'] = float(np.mean(values))
                aggregate_metrics[f'{metric}_std'] = float(np.std(values))
                aggregate_metrics[f'{metric}_min'] = float(np.min(values))
                aggregate_metrics[f'{metric}_max'] = float(np.max(values))
        
        # Find best fold
        best_fold_idx = np.argmax(all_val_metrics['mAP50'])
        best_fold = fold_results[best_fold_idx]
        
        # Final evaluation on test set using best model
        print(f"\n{'='*60}")
        print(f"Evaluating best model (Fold {best_fold['fold']}) on test set")
        print(f"{'='*60}\n")
        
        # Prepare test dataset
        test_splits = {
            'train': [],
            'val': [],
            'test': test_data
        }
        test_yaml = self.data_prep.prepare_yolo_dataset(
            test_splits,
            self.config.DATA_DIR / 'kfold_datasets',
            f"{model_type}_test_set"
        )
        
        # Load and evaluate best model
        model = self.model_manager.get_model(model_type)
        best_model_path = best_fold['train_results']['best_model_path']
        model.load_model(best_model_path)
        test_metrics = model.evaluate(test_yaml, split='test')
        
        # Compile final results
        kfold_results = {
            'model_type': model_type,
            'k_folds': k,
            'total_samples': len(dataset),
            'test_samples': len(test_data),
            'total_training_time': total_time,
            'aggregate_metrics': aggregate_metrics,
            'best_fold': best_fold['fold'],
            'best_model_path': best_model_path,
            'test_metrics': test_metrics,
            'fold_results': fold_results if save_all_models else [best_fold],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = self.config.RESULTS_DIR / f'{model_type}_kfold_results.json'
        with open(results_path, 'w') as f:
            json.dump(kfold_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Complete!")
        print(f"{'='*60}")
        print(f"Mean mAP50: {aggregate_metrics['mAP50_mean']:.4f} ± {aggregate_metrics['mAP50_std']:.4f}")
        print(f"Mean Precision: {aggregate_metrics['precision_mean']:.4f} ± {aggregate_metrics['precision_std']:.4f}")
        print(f"Mean Recall: {aggregate_metrics['recall_mean']:.4f} ± {aggregate_metrics['recall_std']:.4f}")
        print(f"\nBest Fold: {best_fold['fold']}")
        print(f"Test Set mAP50: {test_metrics['mAP50']:.4f}")
        print(f"Test Set Precision: {test_metrics['precision']:.4f}")
        print(f"Test Set Recall: {test_metrics['recall']:.4f}")
        print(f"\nResults saved to: {results_path}")
        
        return kfold_results
    
    def train_standard_split(self, model_type: str, dataset_dir: Path) -> Dict:
        """
        Train with standard train/val/test split
        
        Args:
            model_type: Type of model to train
            dataset_dir: Directory containing the dataset
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type} with Standard Split")
        print(f"Train: {self.config.TRAIN_SPLIT*100}% | Val: {self.config.VAL_SPLIT*100}% | Test: {self.config.TEST_SPLIT*100}%")
        print(f"{'='*60}\n")
        
        # Organize dataset
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        dataset = self.data_prep.organize_dataset(images_dir, labels_dir)
        
        if len(dataset) == 0:
            raise ValueError(f"No data found in {dataset_dir}")
        
        print(f"Total samples: {len(dataset)}")
        
        # Create standard splits
        splits = self.data_prep.split_dataset(
            dataset,
            train_ratio=self.config.TRAIN_SPLIT,
            val_ratio=self.config.VAL_SPLIT,
            test_ratio=self.config.TEST_SPLIT
        )
        
        # Prepare YOLO dataset
        yaml_path = self.data_prep.prepare_yolo_dataset(
            splits,
            self.config.DATA_DIR,
            f"{model_type}_standard"
        )
        
        # Train
        results = self.train_single_fold(model_type, yaml_path)
        
        # Save results
        results_path = self.config.RESULTS_DIR / f'{model_type}_standard_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
        return results
    
    def few_shot_fine_tune(self, base_model_path: str, 
                          few_shot_dir: Path,
                          model_type: str = None) -> Dict:
        """
        Perform few-shot fine-tuning on real photos
        
        Args:
            base_model_path: Path to base model trained on renders
            few_shot_dir: Directory containing real photos for fine-tuning
            model_type: Type of model (inferred from path if None)
            
        Returns:
            Dictionary with fine-tuning results
        """
        print(f"\n{'='*60}")
        print(f"Few-Shot Fine-Tuning")
        print(f"{'='*60}\n")
        
        # Infer model type from path if not provided
        if model_type is None:
            for mtype in self.model_manager.list_models():
                if mtype in base_model_path:
                    model_type = mtype
                    break
            if model_type is None:
                model_type = self.config.DEFAULT_MODEL
        
        # Organize few-shot data
        few_shot_data = self.data_prep.organize_few_shot_data(few_shot_dir)
        
        if len(few_shot_data) == 0:
            raise ValueError(f"No few-shot data found in {few_shot_dir}")
        
        print(f"Few-shot samples: {len(few_shot_data)}")
        
        # Create train/val split (no test for few-shot)
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(
            few_shot_data, 
            test_size=0.2,
            random_state=42
        )
        
        # Prepare dataset
        splits = {
            'train': train_data,
            'val': val_data,
            'test': []
        }
        
        yaml_path = self.data_prep.prepare_yolo_dataset(
            splits,
            self.config.DATA_DIR,
            f"{model_type}_fewshot"
        )
        
        # Fine-tune
        model = self.model_manager.get_model(model_type)
        fine_tune_results = model.fine_tune(
            yaml_path,
            base_model_path=base_model_path,
            name=f"{model_type}_finetuned"
        )
        
        # Evaluate
        print("\nEvaluating fine-tuned model...")
        model.load_model(fine_tune_results['best_model_path'])
        val_metrics = model.evaluate(yaml_path, split='val')
        
        results = {
            'model_type': model_type,
            'base_model': base_model_path,
            'few_shot_samples': len(few_shot_data),
            'timestamp': datetime.now().isoformat(),
            'fine_tune_results': fine_tune_results,
            'val_metrics': val_metrics
        }
        
        # Save results
        results_path = self.config.RESULTS_DIR / f'{model_type}_fewshot_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFine-tuning complete!")
        print(f"Fine-tuned model: {fine_tune_results['best_model_path']}")
        print(f"Validation mAP50: {val_metrics['mAP50']:.4f}")
        print(f"Results saved to: {results_path}")
        
        return results
