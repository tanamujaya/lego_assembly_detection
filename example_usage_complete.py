"""
Example Usage Script for LEGO Assembly Error Detection System
Demonstrates all major features of the system
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from data_preparation import DatasetPreparer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from order_processor import OrderProcessor, OrderRequest, RaspberryPiOptimizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_prepare_dataset():
    """Example 1: Prepare dataset with train/val/test splits"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Dataset Preparation")
    logger.info("="*60)
    
    preparer = DatasetPreparer(config)
    
    # Prepare standard dataset
    dataset_info = preparer.prepare_yolo_dataset(
        images_path="path/to/rendered/images",
        labels_path="path/to/labels",
        output_path="data/prepared_dataset"
    )
    
    logger.info(f"Dataset prepared:")
    logger.info(f"  - Train: {dataset_info['train_size']} images")
    logger.info(f"  - Val: {dataset_info['val_size']} images")
    logger.info(f"  - Test: {dataset_info['test_size']} images")
    logger.info(f"  - YAML: {dataset_info['yaml_path']}")


def example_2_kfold_preparation():
    """Example 2: Create K-fold cross-validation splits"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: K-Fold Cross-Validation Preparation")
    logger.info("="*60)
    
    preparer = DatasetPreparer(config)
    
    # Create K-fold splits
    fold_info = preparer.create_kfold_splits(
        images_path="path/to/rendered/images",
        labels_path="path/to/labels",
        output_path="data/kfold_dataset"
    )
    
    logger.info(f"Created {len(fold_info)} folds")
    for fold in fold_info:
        logger.info(f"  Fold {fold['fold']}: "
                   f"Train={fold['train_size']}, "
                   f"Val={fold['val_size']}, "
                   f"Test={fold['test_size']}")


def example_3_train_standard():
    """Example 3: Standard model training"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Standard Model Training")
    logger.info("="*60)
    
    trainer = ModelTrainer(config)
    
    # Train YOLOv8n model
    results = trainer.train_yolo_model(
        data_yaml="data/prepared_dataset/dataset.yaml",
        save_name="yolov8n_best.pt"
    )
    
    logger.info(f"Training completed:")
    logger.info(f"  - Model: {results['model_type']}")
    logger.info(f"  - Time: {results['training_time']:.2f}s")
    logger.info(f"  - Saved to: {results['best_model_path']}")


def example_4_train_kfold():
    """Example 4: K-fold cross-validation training"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: K-Fold Cross-Validation Training")
    logger.info("="*60)
    
    trainer = ModelTrainer(config)
    
    # Train with K-fold
    fold_results = trainer.train_with_kfold(
        fold_info_path="data/kfold_dataset/fold_info.json"
    )
    
    logger.info(f"K-fold training completed:")
    logger.info(f"  - Number of folds: {len(fold_results)}")
    
    for result in fold_results:
        logger.info(f"  - Fold {result['fold']}: {result['training_time']:.2f}s")


def example_5_few_shot_finetuning():
    """Example 5: Few-shot fine-tuning with real photos"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Few-Shot Fine-Tuning")
    logger.info("="*60)
    
    # First prepare few-shot dataset
    preparer = DatasetPreparer(config)
    
    few_shot_info = preparer.prepare_few_shot_dataset(
        real_photos_path="path/to/real/photos",
        labels_path="path/to/real/labels",
        base_model_path="models/yolov8n_best.pt"
    )
    
    logger.info(f"Few-shot dataset prepared:")
    logger.info(f"  - Total samples: {few_shot_info['total_samples']}")
    logger.info(f"  - Samples per class: {few_shot_info['samples_per_class']}")
    
    # Fine-tune model
    trainer = ModelTrainer(config)
    
    results = trainer.fine_tune_few_shot(
        base_model_path="models/yolov8n_best.pt",
        few_shot_yaml=few_shot_info['yaml_path'],
        save_name="yolov8n_finetuned.pt"
    )
    
    logger.info(f"Fine-tuning completed:")
    logger.info(f"  - Time: {results['finetuning_time']:.2f}s")
    logger.info(f"  - Saved to: {results['finetuned_model_path']}")


def example_6_evaluate_model():
    """Example 6: Comprehensive model evaluation"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 6: Model Evaluation")
    logger.info("="*60)
    
    evaluator = ModelEvaluator(config)
    
    # Evaluate model
    metrics = evaluator.evaluate_model(
        model_path="models/yolov8n_best.pt",
        test_data_yaml="data/prepared_dataset/dataset.yaml",
        save_visualizations=True
    )
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"  - Precision: {metrics['precision']:.4f}")
    logger.info(f"  - Recall: {metrics['recall']:.4f}")
    logger.info(f"  - F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  - mAP@0.5: {metrics['map50']:.4f}")
    logger.info(f"  - mAP@0.5:0.95: {metrics['map50_95']:.4f}")


def example_7_measure_inference_time():
    """Example 7: Measure inference time for Raspberry Pi"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 7: Inference Time Measurement")
    logger.info("="*60)
    
    evaluator = ModelEvaluator(config)
    
    # Get test images
    test_images = [
        "data/prepared_dataset/test/images/image_001.jpg",
        "data/prepared_dataset/test/images/image_002.jpg",
        "data/prepared_dataset/test/images/image_003.jpg",
    ]
    
    # Measure inference time
    timing_stats = evaluator.evaluate_inference_time(
        model_path="models/yolov8n_best.pt",
        test_images=test_images,
        num_iterations=100
    )
    
    logger.info(f"Inference timing:")
    logger.info(f"  - Mean time: {timing_stats['mean_inference_time']:.4f}s")
    logger.info(f"  - Std dev: {timing_stats['std_inference_time']:.4f}s")
    logger.info(f"  - Min time: {timing_stats['min_inference_time']:.4f}s")
    logger.info(f"  - Max time: {timing_stats['max_inference_time']:.4f}s")
    logger.info(f"  - FPS: {timing_stats['fps']:.2f}")


def example_8_compare_models():
    """Example 8: Compare different model architectures"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 8: Model Architecture Comparison")
    logger.info("="*60)
    
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)
    
    model_results = []
    
    # Train and evaluate multiple models
    for model_type in ['yolov8n', 'yolov8s']:
        logger.info(f"\nTraining {model_type}...")
        
        # Train
        train_result = trainer.train_yolo_model(
            data_yaml="data/prepared_dataset/dataset.yaml",
            model_name=model_type,
            save_name=f"{model_type}_best.pt"
        )
        
        # Evaluate
        eval_result = evaluator.evaluate_model(
            model_path=train_result['best_model_path'],
            test_data_yaml="data/prepared_dataset/dataset.yaml",
            save_visualizations=False
        )
        
        model_results.append(eval_result)
    
    # Compare results
    comparison = evaluator.compare_models(model_results)
    
    logger.info(f"\nModel comparison completed:")
    logger.info(f"  - Number of models: {len(comparison['models'])}")


def example_9_process_single_order():
    """Example 9: Process a single order"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 9: Single Order Processing")
    logger.info("="*60)
    
    # Initialize processor
    processor = OrderProcessor(config)
    processor.load_model("models/yolov8n_best.pt")
    
    # Create order
    order = OrderRequest(
        order_id="ORDER_001",
        model_type="LEGO_HOUSE",
        reference_image="data/reference_models/house.jpg"
    )
    
    # Process order
    result = processor.process_order(
        order=order,
        real_photo_path="data/captured_photos/order_001.jpg"
    )
    
    logger.info(f"Order processing result:")
    logger.info(f"  - Order ID: {result.order_id}")
    logger.info(f"  - Decision: {result.decision}")
    logger.info(f"  - Confidence: {result.confidence:.2f}")
    logger.info(f"  - Processing time: {result.processing_time:.3f}s")
    logger.info(f"  - Detected errors: {result.detected_errors}")


def example_10_batch_processing():
    """Example 10: Batch order processing"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 10: Batch Order Processing")
    logger.info("="*60)
    
    # Initialize processor
    processor = OrderProcessor(config)
    processor.load_model("models/yolov8n_best.pt")
    
    # Create batch of orders
    orders = [
        (OrderRequest(order_id=f"ORDER_{i:03d}", model_type="LEGO_HOUSE"),
         f"data/captured_photos/order_{i:03d}.jpg")
        for i in range(1, 11)
    ]
    
    # Process batch
    results = processor.batch_process_orders(orders)
    
    # Print summary
    correct = sum(1 for r in results if r.decision == "RIGHT")
    incorrect = sum(1 for r in results if r.decision == "WRONG")
    avg_time = sum(r.processing_time for r in results) / len(results)
    
    logger.info(f"Batch processing completed:")
    logger.info(f"  - Total orders: {len(results)}")
    logger.info(f"  - Correct: {correct}")
    logger.info(f"  - Incorrect: {incorrect}")
    logger.info(f"  - Average processing time: {avg_time:.3f}s")


def example_11_raspberry_pi_optimization():
    """Example 11: Optimize model for Raspberry Pi"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 11: Raspberry Pi Optimization")
    logger.info("="*60)
    
    optimizer = RaspberryPiOptimizer(config)
    
    # Optimize model
    optimized_path = optimizer.optimize_model(
        model_path="models/yolov8n_best.pt",
        output_path="models/yolov8n_optimized.onnx"
    )
    
    logger.info(f"Model optimized for Raspberry Pi:")
    logger.info(f"  - Optimized model: {optimized_path}")
    
    # Set CPU threads
    optimizer.set_cpu_threads(2)
    
    # Enable memory optimization
    optimizer.enable_memory_optimization()
    
    logger.info(f"  - CPU threads: 2")
    logger.info(f"  - Memory optimization: Enabled")


def example_12_create_order_json():
    """Example 12: Create order from JSON file"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 12: Create Order from JSON")
    logger.info("="*60)
    
    import json
    
    # Create order JSON
    order_data = {
        "order_id": "ORDER_JSON_001",
        "model_type": "LEGO_CASTLE",
        "reference_image": "data/reference_models/castle.jpg",
        "timestamp": "2025-11-06T10:30:00"
    }
    
    order_path = "data/orders/ORDER_JSON_001_order.json"
    with open(order_path, 'w') as f:
        json.dump(order_data, f, indent=2)
    
    logger.info(f"Order JSON created: {order_path}")
    
    # Process order from JSON
    processor = OrderProcessor(config)
    processor.load_model("models/yolov8n_best.pt")
    
    result = processor.process_order_from_json(
        order_json_path=order_path,
        real_photo_path="data/captured_photos/order_json_001.jpg"
    )
    
    logger.info(f"Order processed from JSON:")
    logger.info(f"  - Decision: {result.decision}")
    logger.info(f"  - Confidence: {result.confidence:.2f}")


def example_13_statistics_tracking():
    """Example 13: Track processing statistics"""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 13: Processing Statistics")
    logger.info("="*60)
    
    processor = OrderProcessor(config)
    processor.load_model("models/yolov8n_best.pt")
    
    # Process multiple orders
    for i in range(1, 6):
        order = OrderRequest(order_id=f"STAT_{i:03d}", model_type="LEGO_MODEL")
        result = processor.process_order(
            order=order,
            real_photo_path=f"data/captured_photos/stat_{i:03d}.jpg"
        )
    
    # Get statistics
    stats = processor.get_statistics()
    
    logger.info(f"Processing statistics:")
    logger.info(f"  - Total orders: {stats['total_orders']}")
    logger.info(f"  - Correct assemblies: {stats['correct_assemblies']}")
    logger.info(f"  - Incorrect assemblies: {stats['incorrect_assemblies']}")
    logger.info(f"  - Accuracy: {stats['accuracy']:.2%}")
    logger.info(f"  - Average processing time: {stats['avg_processing_time']:.3f}s")


def main():
    """Run all examples"""
    
    examples = [
        ("Dataset Preparation", example_1_prepare_dataset),
        ("K-Fold Preparation", example_2_kfold_preparation),
        ("Standard Training", example_3_train_standard),
        ("K-Fold Training", example_4_train_kfold),
        ("Few-Shot Fine-Tuning", example_5_few_shot_finetuning),
        ("Model Evaluation", example_6_evaluate_model),
        ("Inference Time Measurement", example_7_measure_inference_time),
        ("Model Comparison", example_8_compare_models),
        ("Single Order Processing", example_9_process_single_order),
        ("Batch Processing", example_10_batch_processing),
        ("Raspberry Pi Optimization", example_11_raspberry_pi_optimization),
        ("Create Order from JSON", example_12_create_order_json),
        ("Statistics Tracking", example_13_statistics_tracking),
    ]
    
    logger.info("\n" + "="*60)
    logger.info("LEGO ASSEMBLY ERROR DETECTION - EXAMPLES")
    logger.info("="*60)
    logger.info(f"\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        logger.info(f"  {i}. {name}")
    
    logger.info("\n" + "="*60)
    logger.info("To run a specific example, call it from Python:")
    logger.info("  from example_usage import example_1_prepare_dataset")
    logger.info("  example_1_prepare_dataset()")
    logger.info("="*60)


if __name__ == "__main__":
    main()
