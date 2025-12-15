"""
Data Preparation Module for LEGO Assembly Error Detection
Handles dataset splitting, K-fold cross-validation, and data loading
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold, train_test_split
import yaml
import logging

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepares datasets with train/val/test splits and K-fold cross-validation"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = config.DATA_DIR
        self.dataset_config = config.DATASET_CONFIG
        self.random_seed = self.dataset_config['random_seed']
        
    def prepare_yolo_dataset(self, 
                            images_path: str, 
                            labels_path: str,
                            output_path: str = None) -> Dict:
        """
        Prepare YOLO format dataset with train/val/test splits
        
        Args:
            images_path: Path to images directory
            labels_path: Path to labels directory (YOLO format .txt files)
            output_path: Output directory for organized dataset
            
        Returns:
            Dictionary with dataset information
        """
        if output_path is None:
            output_path = self.data_dir / "prepared_dataset"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        images_path = Path(images_path)
        labels_path = Path(labels_path)
        
        image_files = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in valid_extensions:
            image_files.extend(list(images_path.glob(f"*{ext}")))
            image_files.extend(list(images_path.glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_files)} images")
        
        # Filter images that have corresponding labels
        valid_data = []
        for img_path in image_files:
            label_path = labels_path / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_data.append((img_path, label_path))
        
        logger.info(f"Found {len(valid_data)} valid image-label pairs")
        
        if len(valid_data) == 0:
            raise ValueError("No valid image-label pairs found!")
        
        # Shuffle data
        np.random.seed(self.random_seed)
        np.random.shuffle(valid_data)
        
        # Split into train, val, test
        train_split = self.dataset_config['train_split']
        val_split = self.dataset_config['val_split']
        test_split = self.dataset_config['test_split']
        
        n_total = len(valid_data)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val
        
        train_data = valid_data[:n_train]
        val_data = valid_data[n_train:n_train + n_val]
        test_data = valid_data[n_train + n_val:]
        
        logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Create directory structure
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            # Create directories
            split_images = output_path / split_name / 'images'
            split_labels = output_path / split_name / 'labels'
            split_images.mkdir(parents=True, exist_ok=True)
            split_labels.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for img_path, label_path in split_data:
                shutil.copy2(img_path, split_images / img_path.name)
                shutil.copy2(label_path, split_labels / label_path.name)
        
        # Create YAML configuration file for YOLO
        yaml_config = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.config.MODEL_CONFIG['num_classes'],
            'names': list(self.config.CLASS_NAMES.values())
        }
        
        yaml_path = output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        logger.info(f"Dataset prepared and saved to {output_path}")
        logger.info(f"YAML config saved to {yaml_path}")
        
        return {
            'output_path': str(output_path),
            'yaml_path': str(yaml_path),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'total_size': n_total
        }
    
    def create_kfold_splits(self, 
                           images_path: str, 
                           labels_path: str,
                           output_path: str = None) -> List[Dict]:
        """
        Create K-fold cross-validation splits
        
        Args:
            images_path: Path to images directory
            labels_path: Path to labels directory
            output_path: Output directory for K-fold splits
            
        Returns:
            List of dictionaries with fold information
        """
        if output_path is None:
            output_path = self.data_dir / "kfold_dataset"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all valid image-label pairs
        images_path = Path(images_path)
        labels_path = Path(labels_path)
        
        image_files = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in valid_extensions:
            image_files.extend(list(images_path.glob(f"*{ext}")))
            image_files.extend(list(images_path.glob(f"*{ext.upper()}")))
        
        valid_data = []
        for img_path in image_files:
            label_path = labels_path / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_data.append((img_path, label_path))
        
        logger.info(f"Creating {self.dataset_config['k_folds']}-fold splits from {len(valid_data)} samples")
        
        # Prepare K-fold
        kf = KFold(n_splits=self.dataset_config['k_folds'], 
                   shuffle=True, 
                   random_state=self.random_seed)
        
        fold_info = []
        valid_data = np.array(valid_data, dtype=object)
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(valid_data)):
            fold_path = output_path / f"fold_{fold_idx + 1}"
            fold_path.mkdir(parents=True, exist_ok=True)
            
            # Get train+val and test data
            train_val_data = valid_data[train_val_idx]
            test_data = valid_data[test_idx]
            
            # Further split train_val into train and val (preserving original ratio)
            val_ratio = self.dataset_config['val_split'] / (
                self.dataset_config['train_split'] + self.dataset_config['val_split']
            )
            
            train_data, val_data = train_test_split(
                train_val_data, 
                test_size=val_ratio, 
                random_state=self.random_seed
            )
            
            # Create directory structure for this fold
            splits = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            for split_name, split_data in splits.items():
                split_images = fold_path / split_name / 'images'
                split_labels = fold_path / split_name / 'labels'
                split_images.mkdir(parents=True, exist_ok=True)
                split_labels.mkdir(parents=True, exist_ok=True)
                
                for img_path, label_path in split_data:
                    shutil.copy2(img_path, split_images / img_path.name)
                    shutil.copy2(label_path, split_labels / label_path.name)
            
            # Create YAML for this fold
            yaml_config = {
                'path': str(fold_path.absolute()),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': self.config.MODEL_CONFIG['num_classes'],
                'names': list(self.config.CLASS_NAMES.values())
            }
            
            yaml_path = fold_path / 'dataset.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False)
            
            fold_info.append({
                'fold': fold_idx + 1,
                'fold_path': str(fold_path),
                'yaml_path': str(yaml_path),
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data)
            })
            
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_data)}, "
                       f"Val={len(val_data)}, Test={len(test_data)}")
        
        # Save fold information
        fold_info_path = output_path / 'fold_info.json'
        with open(fold_info_path, 'w') as f:
            json.dump(fold_info, f, indent=2)
        
        logger.info(f"K-fold splits created and saved to {output_path}")
        
        return fold_info
    
    def prepare_few_shot_dataset(self,
                                real_photos_path: str,
                                labels_path: str,
                                base_model_path: str,
                                output_path: str = None) -> Dict:
        """
        Prepare few-shot learning dataset with real photos
        
        Args:
            real_photos_path: Path to real photo directory
            labels_path: Path to labels for real photos
            base_model_path: Path to base trained model
            output_path: Output directory for few-shot dataset
            
        Returns:
            Dictionary with few-shot dataset information
        """
        if output_path is None:
            output_path = self.data_dir / "few_shot_dataset"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        real_photos_path = Path(real_photos_path)
        labels_path = Path(labels_path)
        
        # Get all real photos with labels
        image_files = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in valid_extensions:
            image_files.extend(list(real_photos_path.glob(f"*{ext}")))
            image_files.extend(list(real_photos_path.glob(f"*{ext.upper()}")))
        
        valid_data = []
        for img_path in image_files:
            label_path = labels_path / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_data.append((img_path, label_path))
        
        logger.info(f"Found {len(valid_data)} real photos with labels")
        
        # Group by class
        class_data = {i: [] for i in range(self.config.MODEL_CONFIG['num_classes'])}
        
        for img_path, label_path in valid_data:
            # Read label to determine class
            with open(label_path, 'r') as f:
                label_content = f.read().strip()
                if label_content:
                    class_id = int(label_content.split()[0])
                    class_data[class_id].append((img_path, label_path))
        
        # Sample few-shot examples
        num_shots = self.config.FEW_SHOT_CONFIG['num_shots']
        few_shot_data = []
        
        for class_id, data in class_data.items():
            if len(data) < num_shots:
                logger.warning(f"Class {class_id} has only {len(data)} samples, "
                             f"need {num_shots} for few-shot learning")
                sampled = data
            else:
                np.random.seed(self.random_seed)
                sampled = [data[i] for i in np.random.choice(
                    len(data), num_shots, replace=False)]
            
            few_shot_data.extend(sampled)
            logger.info(f"Class {class_id} ({self.config.CLASS_NAMES[class_id]}): "
                       f"{len(sampled)} samples")
        
        # Create directory structure
        images_dir = output_path / 'images'
        labels_dir = output_path / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img_path, label_path in few_shot_data:
            shutil.copy2(img_path, images_dir / img_path.name)
            shutil.copy2(label_path, labels_dir / label_path.name)
        
        # Create YAML configuration
        yaml_config = {
            'path': str(output_path.absolute()),
            'train': 'images',
            'val': 'images',  # Use same for validation in few-shot
            'nc': self.config.MODEL_CONFIG['num_classes'],
            'names': list(self.config.CLASS_NAMES.values())
        }
        
        yaml_path = output_path / 'few_shot.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        few_shot_info = {
            'output_path': str(output_path),
            'yaml_path': str(yaml_path),
            'total_samples': len(few_shot_data),
            'samples_per_class': {
                self.config.CLASS_NAMES[i]: len([d for d in few_shot_data 
                                                if self._get_class_from_label(d[1]) == i])
                for i in range(self.config.MODEL_CONFIG['num_classes'])
            },
            'base_model': base_model_path
        }
        
        logger.info(f"Few-shot dataset prepared with {len(few_shot_data)} samples")
        
        return few_shot_info
    
    def _get_class_from_label(self, label_path: Path) -> int:
        """Extract class ID from YOLO label file"""
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
            if label_content:
                return int(label_content.split()[0])
        return -1
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """
        Validate dataset structure and contents
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            True if valid, False otherwise
        """
        dataset_path = Path(dataset_path)
        
        # Check if YAML exists
        yaml_path = dataset_path / 'dataset.yaml'
        if not yaml_path.exists():
            logger.error(f"YAML config not found at {yaml_path}")
            return False
        
        # Load YAML
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Check required splits
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            if split not in yaml_config:
                logger.error(f"Split '{split}' not found in YAML config")
                return False
            
            # Check if directories exist
            images_dir = dataset_path / yaml_config[split]
            labels_dir = dataset_path / yaml_config[split].replace('images', 'labels')
            
            if not images_dir.exists():
                logger.error(f"Images directory not found: {images_dir}")
                return False
            
            if not labels_dir.exists():
                logger.error(f"Labels directory not found: {labels_dir}")
                return False
            
            # Check if files exist
            image_files = list(images_dir.iterdir())
            label_files = list(labels_dir.iterdir())
            
            if len(image_files) == 0:
                logger.error(f"No images found in {images_dir}")
                return False
            
            if len(label_files) == 0:
                logger.error(f"No labels found in {labels_dir}")
                return False
        
        logger.info(f"Dataset validation passed for {dataset_path}")
        return True
