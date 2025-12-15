"""
Evaluation and Metrics Visualization Module
Generates comprehensive performance reports with visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime

from config import Config


class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.config.setup_directories()
        sns.set_style("whitegrid")
        
    def load_results(self, results_path: Path) -> Dict:
        """Load training/evaluation results from JSON"""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_kfold_metrics(self, kfold_results: Dict, 
                          save_path: Path = None) -> Path:
        """
        Plot K-fold cross-validation metrics
        
        Args:
            kfold_results: Results from K-fold training
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        # Extract fold metrics
        fold_numbers = []
        precision_scores = []
        recall_scores = []
        map50_scores = []
        map_scores = []
        
        for fold_result in kfold_results['fold_results']:
            fold_numbers.append(fold_result['fold'])
            metrics = fold_result['val_metrics']
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
            map50_scores.append(metrics['mAP50'])
            map_scores.append(metrics['mAP50-95'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{kfold_results["model_type"]} - K-Fold Cross-Validation Results', 
                    fontsize=16, fontweight='bold')
        
        # Plot Precision
        axes[0, 0].plot(fold_numbers, precision_scores, marker='o', linewidth=2, 
                       markersize=8, color='#2E86AB')
        axes[0, 0].axhline(y=np.mean(precision_scores), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(precision_scores):.4f}')
        axes[0, 0].fill_between(fold_numbers, 
                               np.mean(precision_scores) - np.std(precision_scores),
                               np.mean(precision_scores) + np.std(precision_scores),
                               alpha=0.2, color='#2E86AB')
        axes[0, 0].set_xlabel('Fold', fontsize=12)
        axes[0, 0].set_ylabel('Precision', fontsize=12)
        axes[0, 0].set_title('Precision across Folds', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Recall
        axes[0, 1].plot(fold_numbers, recall_scores, marker='s', linewidth=2, 
                       markersize=8, color='#A23B72')
        axes[0, 1].axhline(y=np.mean(recall_scores), color='r', linestyle='--',
                          label=f'Mean: {np.mean(recall_scores):.4f}')
        axes[0, 1].fill_between(fold_numbers,
                               np.mean(recall_scores) - np.std(recall_scores),
                               np.mean(recall_scores) + np.std(recall_scores),
                               alpha=0.2, color='#A23B72')
        axes[0, 1].set_xlabel('Fold', fontsize=12)
        axes[0, 1].set_ylabel('Recall', fontsize=12)
        axes[0, 1].set_title('Recall across Folds', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot mAP50
        axes[1, 0].plot(fold_numbers, map50_scores, marker='^', linewidth=2, 
                       markersize=8, color='#F18F01')
        axes[1, 0].axhline(y=np.mean(map50_scores), color='r', linestyle='--',
                          label=f'Mean: {np.mean(map50_scores):.4f}')
        axes[1, 0].fill_between(fold_numbers,
                               np.mean(map50_scores) - np.std(map50_scores),
                               np.mean(map50_scores) + np.std(map50_scores),
                               alpha=0.2, color='#F18F01')
        axes[1, 0].set_xlabel('Fold', fontsize=12)
        axes[1, 0].set_ylabel('mAP50', fontsize=12)
        axes[1, 0].set_title('mAP@0.5 across Folds', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot mAP50-95
        axes[1, 1].plot(fold_numbers, map_scores, marker='D', linewidth=2, 
                       markersize=8, color='#6A994E')
        axes[1, 1].axhline(y=np.mean(map_scores), color='r', linestyle='--',
                          label=f'Mean: {np.mean(map_scores):.4f}')
        axes[1, 1].fill_between(fold_numbers,
                               np.mean(map_scores) - np.std(map_scores),
                               np.mean(map_scores) + np.std(map_scores),
                               alpha=0.2, color='#6A994E')
        axes[1, 1].set_xlabel('Fold', fontsize=12)
        axes[1, 1].set_ylabel('mAP50-95', fontsize=12)
        axes[1, 1].set_title('mAP@0.5:0.95 across Folds', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.config.RESULTS_DIR / f'{kfold_results["model_type"]}_kfold_metrics.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"K-fold metrics plot saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(self, results_list: List[Dict], 
                               save_path: Path = None) -> Path:
        """
        Compare metrics across different models or training runs
        
        Args:
            results_list: List of result dictionaries
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        # Extract metrics
        model_names = []
        precision = []
        recall = []
        map50 = []
        map_scores = []
        
        for result in results_list:
            # Handle different result formats
            if 'aggregate_metrics' in result:  # K-fold results
                model_names.append(result['model_type'])
                precision.append(result['aggregate_metrics']['precision_mean'])
                recall.append(result['aggregate_metrics']['recall_mean'])
                map50.append(result['aggregate_metrics']['mAP50_mean'])
                map_scores.append(result['aggregate_metrics']['mAP50-95_mean'])
            elif 'test_metrics' in result:  # Standard results
                model_names.append(result['model_type'])
                precision.append(result['test_metrics']['precision'])
                recall.append(result['test_metrics']['recall'])
                map50.append(result['test_metrics']['mAP50'])
                map_scores.append(result['test_metrics']['mAP50-95'])
        
        # Create bar plot
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars1 = ax.bar(x - 1.5*width, precision, width, label='Precision', color='#2E86AB')
        bars2 = ax.bar(x - 0.5*width, recall, width, label='Recall', color='#A23B72')
        bars3 = ax.bar(x + 0.5*width, map50, width, label='mAP50', color='#F18F01')
        bars4 = ax.bar(x + 1.5*width, map_scores, width, label='mAP50-95', color='#6A994E')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'model_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_confusion_matrix(self, y_true: List, y_pred: List,
                             class_names: List[str] = None,
                             save_path: Path = None) -> Path:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        from sklearn.metrics import confusion_matrix
        
        if class_names is None:
            class_names = self.config.CLASSES
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'confusion_matrix.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def plot_inference_time_distribution(self, inspection_results: List[Dict],
                                        save_path: Path = None) -> Path:
        """
        Plot distribution of inference times
        
        Args:
            inspection_results: List of inspection result dictionaries
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        inference_times = [r['assessment_time'] for r in inspection_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(inference_times, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(inference_times), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(inference_times):.3f}s')
        ax1.axvline(np.median(inference_times), color='g', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(inference_times):.3f}s')
        ax1.set_xlabel('Inference Time (seconds)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Inference Time Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(inference_times, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Inference Time (seconds)', fontsize=12)
        ax2.set_title('Inference Time Statistics', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f'Min: {np.min(inference_times):.3f}s\n'
        stats_text += f'Max: {np.max(inference_times):.3f}s\n'
        stats_text += f'Std: {np.std(inference_times):.3f}s'
        ax2.text(1.15, np.median(inference_times), stats_text,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.config.RESULTS_DIR / 'inference_time_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inference time distribution plot saved to: {save_path}")
        plt.close()
        
        return save_path
    
    def generate_evaluation_report(self, results: Dict, 
                                  output_path: Path = None) -> Path:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Training/evaluation results
            output_path: Path to save report
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.config.RESULTS_DIR / f'evaluation_report_{timestamp}.txt'
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LEGO ASSEMBLY ERROR DETECTION - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {results['model_type']}\n\n")
            
            # K-fold results
            if 'aggregate_metrics' in results:
                f.write("-"*70 + "\n")
                f.write("K-FOLD CROSS-VALIDATION RESULTS\n")
                f.write("-"*70 + "\n")
                f.write(f"Number of Folds: {results['k_folds']}\n")
                f.write(f"Total Training Time: {results['total_training_time']:.2f} seconds\n\n")
                
                agg = results['aggregate_metrics']
                f.write("Aggregate Metrics:\n")
                f.write(f"  Precision:  {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}\n")
                f.write(f"  Recall:     {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}\n")
                f.write(f"  mAP50:      {agg['mAP50_mean']:.4f} ± {agg['mAP50_std']:.4f}\n")
                f.write(f"  mAP50-95:   {agg['mAP50-95_mean']:.4f} ± {agg['mAP50-95_std']:.4f}\n\n")
                
                f.write(f"Best Fold: {results['best_fold']}\n")
                f.write(f"Best Model Path: {results['best_model_path']}\n\n")
                
                if 'test_metrics' in results:
                    f.write("Test Set Performance:\n")
                    test = results['test_metrics']
                    f.write(f"  Precision: {test['precision']:.4f}\n")
                    f.write(f"  Recall:    {test['recall']:.4f}\n")
                    f.write(f"  mAP50:     {test['mAP50']:.4f}\n")
                    f.write(f"  mAP50-95:  {test['mAP50-95']:.4f}\n")
            
            # Standard split results
            elif 'test_metrics' in results:
                f.write("-"*70 + "\n")
                f.write("TEST SET PERFORMANCE\n")
                f.write("-"*70 + "\n")
                test = results['test_metrics']
                f.write(f"Precision: {test['precision']:.4f}\n")
                f.write(f"Recall:    {test['recall']:.4f}\n")
                f.write(f"mAP50:     {test['mAP50']:.4f}\n")
                f.write(f"mAP50-95:  {test['mAP50-95']:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"Evaluation report saved to: {output_path}")
        return output_path
    
    def create_summary_dashboard(self, results_list: List[Dict]) -> Path:
        """
        Create a comprehensive summary dashboard
        
        Args:
            results_list: List of result dictionaries from different models
            
        Returns:
            Path to saved dashboard
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Extract data
        model_names = []
        metrics_data = {'precision': [], 'recall': [], 'mAP50': [], 'mAP50-95': []}
        
        for result in results_list:
            model_names.append(result['model_type'])
            if 'test_metrics' in result:
                m = result['test_metrics']
            elif 'aggregate_metrics' in result:
                m = {k.replace('_mean', ''): v for k, v in result['aggregate_metrics'].items() 
                     if '_mean' in k}
            else:
                continue
                
            for key in metrics_data.keys():
                if key in m:
                    metrics_data[key].append(m[key])
        
        # Main title
        fig.suptitle('LEGO Assembly Error Detection - Performance Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Precision comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(model_names, metrics_data['precision'], color='#2E86AB')
        ax1.set_title('Precision', fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Recall comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(model_names, metrics_data['recall'], color='#A23B72')
        ax2.set_title('Recall', fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: mAP50 comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(model_names, metrics_data['mAP50'], color='#F18F01')
        ax3.set_title('mAP@0.5', fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Radar chart
        ax4 = fig.add_subplot(gs[1, :], projection='polar')
        angles = np.linspace(0, 2*np.pi, len(metrics_data), endpoint=False).tolist()
        
        for i, model in enumerate(model_names):
            values = [metrics_data[k][i] for k in metrics_data.keys()]
            values += values[:1]  # Complete the circle
            angles_plot = angles + angles[:1]
            ax4.plot(angles_plot, values, 'o-', linewidth=2, label=model)
            ax4.fill(angles_plot, values, alpha=0.25)
        
        ax4.set_xticks(angles)
        ax4.set_xticklabels(list(metrics_data.keys()))
        ax4.set_ylim(0, 1)
        ax4.set_title('Comprehensive Metrics Comparison', fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
        
        # Plot 5: Summary table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        table_data = []
        for i, model in enumerate(model_names):
            row = [model]
            for key in metrics_data.keys():
                row.append(f"{metrics_data[key][i]:.4f}")
            table_data.append(row)
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Model'] + list(metrics_data.keys()),
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(metrics_data) + 1):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        save_path = self.config.RESULTS_DIR / 'performance_dashboard.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
        plt.close()
        
        return save_path
