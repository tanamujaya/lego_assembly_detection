"""
Model Evaluation Module for LEGO Assembly Error Detection
Calculates Precision, Recall, mAP and generates visualizations
"""

import os
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available")

try:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, 
        confusion_matrix, classification_report,
        average_precision_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_config = config.METRICS_CONFIG
        self.results_dir = config.RESULTS_DIR
        
        # Create evaluation directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.results_dir / f"evaluation_{timestamp}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = defaultdict(list)
        
    def evaluate_model(self, 
                      model_path: str,
                      test_data_yaml: str,
                      save_visualizations: bool = True) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model_path: Path to trained model
            test_data_yaml: Path to test dataset YAML
            save_visualizations: Whether to save plots
            
        Returns:
            Dictionary with all metrics
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not installed")
        
        logger.info(f"Evaluating model: {model_path}")
        logger.info(f"Test data: {test_data_yaml}")
        
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        start_time = time.time()
        results = model.val(
            data=test_data_yaml,
            split='test',
            imgsz=self.config.DATASET_CONFIG['image_size'][0],
            batch=self.config.TRAINING_CONFIG['batch_size'],
            device=self.config.TRAINING_CONFIG['device'],
            save_json=True,
            save_hybrid=False,
            conf=self.config.MODEL_CONFIG['confidence_threshold'],
            iou=self.config.MODEL_CONFIG['iou_threshold'],
            plots=True,
            verbose=True
        )
        eval_time = time.time() - start_time
        
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # Extract metrics
        metrics = self._extract_yolo_metrics(results)
        metrics['evaluation_time'] = eval_time
        metrics['model_path'] = model_path
        
        # Calculate additional metrics
        if hasattr(results, 'box'):
            # Per-class metrics
            metrics['per_class_metrics'] = self._calculate_per_class_metrics(results)
        
        # Save metrics
        metrics_path = self.eval_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Generate visualizations
        if save_visualizations:
            self._generate_visualizations(metrics, results)
        
        # Print summary
        self._print_metrics_summary(metrics)
        
        return metrics
    
    def evaluate_inference_time(self,
                               model_path: str,
                               test_images: List[str],
                               num_iterations: int = 100) -> Dict:
        """
        Measure inference time on Raspberry Pi
        
        Args:
            model_path: Path to trained model
            test_images: List of test image paths
            num_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not installed")
        
        logger.info(f"Measuring inference time over {num_iterations} iterations")
        
        model = YOLO(model_path)
        
        inference_times = []
        
        # Warm-up runs
        for _ in range(5):
            _ = model(test_images[0], verbose=False)
        
        # Timed runs
        for i in range(num_iterations):
            img_idx = i % len(test_images)
            
            start_time = time.time()
            _ = model(test_images[img_idx], verbose=False)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
        
        # Calculate statistics
        timing_stats = {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'median_inference_time': np.median(inference_times),
            'fps': 1.0 / np.mean(inference_times),
            'num_iterations': num_iterations,
            'num_test_images': len(test_images)
        }
        
        logger.info(f"Average inference time: {timing_stats['mean_inference_time']:.4f}s")
        logger.info(f"FPS: {timing_stats['fps']:.2f}")
        
        # Check against threshold
        max_time = self.config.PERFORMANCE_THRESHOLDS['max_inference_time']
        if timing_stats['mean_inference_time'] > max_time:
            logger.warning(f"Inference time exceeds threshold: "
                         f"{timing_stats['mean_inference_time']:.4f}s > {max_time}s")
        else:
            logger.info(f"Inference time meets threshold requirement")
        
        # Save timing results
        timing_path = self.eval_dir / 'inference_timing.json'
        with open(timing_path, 'w') as f:
            json.dump(timing_stats, f, indent=2)
        
        # Plot timing distribution
        self._plot_inference_timing(inference_times)
        
        return timing_stats
    
    def compare_models(self, model_results: List[Dict]) -> Dict:
        """
        Compare multiple models
        
        Args:
            model_results: List of evaluation results from different models
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(model_results)} models")
        
        comparison = {
            'models': [],
            'metrics_comparison': {}
        }
        
        # Collect metrics for comparison
        metric_keys = ['precision', 'recall', 'map50', 'map50_95', 
                      'evaluation_time', 'mean_inference_time']
        
        for key in metric_keys:
            comparison['metrics_comparison'][key] = []
        
        for result in model_results:
            model_name = Path(result.get('model_path', 'unknown')).stem
            comparison['models'].append(model_name)
            
            for key in metric_keys:
                value = result.get(key, None)
                if value is not None:
                    comparison['metrics_comparison'][key].append(value)
                else:
                    comparison['metrics_comparison'][key].append(0)
        
        # Generate comparison plots
        self._plot_model_comparison(comparison)
        
        # Save comparison
        comparison_path = self.eval_dir / 'model_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Model comparison saved to {comparison_path}")
        
        return comparison
    
    def _extract_yolo_metrics(self, results) -> Dict:
        """Extract metrics from YOLO validation results"""
        metrics = {}
        
        try:
            if hasattr(results, 'box'):
                box_metrics = results.box
                
                # Overall metrics
                metrics['precision'] = float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0
                metrics['recall'] = float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0
                metrics['map50'] = float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0
                metrics['map50_95'] = float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                     (metrics['precision'] + metrics['recall'] + 1e-6)
                
                # Per-class metrics
                if hasattr(box_metrics, 'ap_class_index'):
                    metrics['ap_per_class'] = {}
                    for idx, ap in zip(box_metrics.ap_class_index, box_metrics.ap):
                        class_name = self.config.CLASS_NAMES.get(int(idx), f'class_{idx}')
                        metrics['ap_per_class'][class_name] = float(ap)
            
            # Speed metrics
            if hasattr(results, 'speed'):
                speed = results.speed
                metrics['preprocess_time'] = speed.get('preprocess', 0)
                metrics['inference_time'] = speed.get('inference', 0)
                metrics['postprocess_time'] = speed.get('postprocess', 0)
                metrics['total_time'] = sum([
                    metrics['preprocess_time'],
                    metrics['inference_time'],
                    metrics['postprocess_time']
                ])
        
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
        
        return metrics
    
    def _calculate_per_class_metrics(self, results) -> Dict:
        """Calculate detailed per-class metrics"""
        per_class = {}
        
        try:
            box_metrics = results.box
            
            if hasattr(box_metrics, 'ap_class_index'):
                for idx in box_metrics.ap_class_index:
                    class_name = self.config.CLASS_NAMES.get(int(idx), f'class_{idx}')
                    
                    # Find metrics for this class
                    class_idx_pos = list(box_metrics.ap_class_index).index(idx)
                    
                    per_class[class_name] = {
                        'precision': float(box_metrics.p[class_idx_pos]) if hasattr(box_metrics, 'p') else 0.0,
                        'recall': float(box_metrics.r[class_idx_pos]) if hasattr(box_metrics, 'r') else 0.0,
                        'ap50': float(box_metrics.ap50[class_idx_pos]) if hasattr(box_metrics, 'ap50') else 0.0,
                        'ap': float(box_metrics.ap[class_idx_pos]) if hasattr(box_metrics, 'ap') else 0.0,
                    }
        
        except Exception as e:
            logger.error(f"Error calculating per-class metrics: {e}")
        
        return per_class
    
    def _generate_visualizations(self, metrics: Dict, results):
        """Generate visualization plots"""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Metrics bar chart
        self._plot_metrics_bars(metrics)
        
        # 2. Per-class performance
        if 'per_class_metrics' in metrics:
            self._plot_per_class_metrics(metrics['per_class_metrics'])
        
        # 3. Confusion matrix (if available)
        if hasattr(results, 'confusion_matrix'):
            self._plot_confusion_matrix(results.confusion_matrix)
        
        logger.info(f"Visualizations saved to {self.eval_dir}")
    
    def _plot_metrics_bars(self, metrics: Dict):
        """Plot main metrics as bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5', 'mAP@0.5:0.95']
        metric_values = [
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('map50', 0),
            metrics.get('map50_95', 0)
        ]
        
        bars = ax.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'metrics_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, per_class_metrics: Dict):
        """Plot per-class performance comparison"""
        if not per_class_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = list(per_class_metrics.keys())
        metrics_to_plot = ['precision', 'recall', 'ap50']
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [per_class_metrics[cls].get(metric, 0) for cls in classes]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm_data):
        """Plot confusion matrix"""
        try:
            if hasattr(cm_data, 'matrix'):
                cm = cm_data.matrix
            else:
                cm = cm_data
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                       xticklabels=list(self.config.CLASS_NAMES.values()),
                       yticklabels=list(self.config.CLASS_NAMES.values()),
                       ax=ax, cbar_kws={'label': 'Count'})
            
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.eval_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            logger.warning(f"Could not plot confusion matrix: {e}")
    
    def _plot_inference_timing(self, inference_times: List[float]):
        """Plot inference timing distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(inference_times, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(inference_times), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(inference_times):.4f}s')
        ax1.set_xlabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Inference Time Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time series
        ax2.plot(inference_times, color='blue', alpha=0.6, linewidth=1)
        ax2.axhline(np.mean(inference_times), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(inference_times):.4f}s')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Inference Time Over Iterations', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'inference_timing.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, comparison: Dict):
        """Plot comparison between multiple models"""
        models = comparison['models']
        metrics_comp = comparison['metrics_comparison']
        
        # Select key metrics for comparison
        key_metrics = ['precision', 'recall', 'map50', 'map50_95']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(key_metrics):
            if metric in metrics_comp:
                values = metrics_comp[metric]
                
                bars = axes[idx].bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
                axes[idx].set_ylabel('Score', fontsize=11, fontweight='bold')
                axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
                axes[idx].set_xticklabels(models, rotation=45, ha='right')
                axes[idx].grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.3f}',
                                 ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_metrics_summary(self, metrics: Dict):
        """Print formatted metrics summary"""
        logger.info("\n" + "="*60)
        logger.info("EVALUATION METRICS SUMMARY")
        logger.info("="*60)
        logger.info(f"Precision:     {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall:        {metrics.get('recall', 0):.4f}")
        logger.info(f"F1-Score:      {metrics.get('f1_score', 0):.4f}")
        logger.info(f"mAP@0.5:       {metrics.get('map50', 0):.4f}")
        logger.info(f"mAP@0.5:0.95:  {metrics.get('map50_95', 0):.4f}")
        logger.info("-"*60)
        logger.info(f"Inference Time: {metrics.get('inference_time', 0):.4f} ms")
        logger.info(f"Total Time:     {metrics.get('total_time', 0):.4f} ms")
        logger.info("="*60 + "\n")
