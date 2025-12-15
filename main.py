"""
Main Script for LEGO Assembly Error Detection System
Complete pipeline: Data preparation, Training, Evaluation, and Inference
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Import custom modules
import config
from data_preparation import DatasetPreparer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from order_processor import OrderProcessor, RaspberryPiOptimizer, OrderRequest

# Setup logging
def setup_logging(log_dir: Path):
    """Configure logging for the system"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def prepare_dataset(args, logger):
    """Prepare dataset with train/val/test splits"""
    logger.info("="*60)
    logger.info("STEP 1: DATASET PREPARATION")
    logger.info("="*60)
    
    preparer = DatasetPreparer(config)
    
    # Prepare standard dataset
    logger.info("Preparing standard dataset...")
    dataset_info = preparer.prepare_yolo_dataset(
        images_path=args.images_path,
        labels_path=args.labels_path,
        output_path=args.output_path
    )
    
    # Validate dataset
    if preparer.validate_dataset(dataset_info['output_path']):
        logger.info("Dataset validation passed ✓")
    else:
        logger.error("Dataset validation failed!")
        sys.exit(1)
    
    return dataset_info


def prepare_kfold_dataset(args, logger):
    """Prepare K-fold cross-validation splits"""
    logger.info("="*60)
    logger.info("STEP 2: K-FOLD DATASET PREPARATION")
    logger.info("="*60)
    
    preparer = DatasetPreparer(config)
    
    logger.info(f"Creating {config.DATASET_CONFIG['k_folds']}-fold splits...")
    fold_info = preparer.create_kfold_splits(
        images_path=args.images_path,
        labels_path=args.labels_path,
        output_path=args.kfold_output
    )
    
    logger.info(f"Created {len(fold_info)} folds successfully ✓")
    
    return fold_info


def train_model(args, logger):
    """Train model on prepared dataset"""
    logger.info("="*60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*60)
    
    trainer = ModelTrainer(config)
    
    if args.use_kfold:
        logger.info("Training with K-fold cross-validation...")
        fold_info_path = Path(args.kfold_output) / 'fold_info.json'
        
        if not fold_info_path.exists():
            logger.error(f"Fold info not found: {fold_info_path}")
            logger.info("Please run dataset preparation with K-fold first")
            sys.exit(1)
        
        results = trainer.train_with_kfold(str(fold_info_path))
        best_model = results[0]['best_model_path']  # Use first fold model
        
    else:
        logger.info("Training single model...")
        dataset_yaml = Path(args.dataset_path) / 'dataset.yaml'
        
        if not dataset_yaml.exists():
            logger.error(f"Dataset YAML not found: {dataset_yaml}")
            sys.exit(1)
        
        results = trainer.train_yolo_model(
            data_yaml=str(dataset_yaml),
            save_name=args.model_name
        )
        best_model = results['best_model_path']
    
    logger.info(f"Training completed ✓")
    logger.info(f"Best model saved: {best_model}")
    
    return best_model


def fine_tune_few_shot(args, logger):
    """Fine-tune model with few-shot learning"""
    logger.info("="*60)
    logger.info("STEP 4: FEW-SHOT FINE-TUNING")
    logger.info("="*60)
    
    if not args.real_photos_path or not Path(args.real_photos_path).exists():
        logger.warning("Real photos not provided, skipping few-shot fine-tuning")
        return args.model_path
    
    # Prepare few-shot dataset
    preparer = DatasetPreparer(config)
    logger.info("Preparing few-shot dataset...")
    
    few_shot_info = preparer.prepare_few_shot_dataset(
        real_photos_path=args.real_photos_path,
        labels_path=args.real_labels_path,
        base_model_path=args.model_path
    )
    
    # Fine-tune model
    trainer = ModelTrainer(config)
    logger.info("Fine-tuning model...")
    
    results = trainer.fine_tune_few_shot(
        base_model_path=args.model_path,
        few_shot_yaml=few_shot_info['yaml_path'],
        save_name='few_shot_finetuned.pt'
    )
    
    logger.info(f"Fine-tuning completed ✓")
    logger.info(f"Fine-tuned model: {results['finetuned_model_path']}")
    
    return results['finetuned_model_path']


def evaluate_model(args, logger):
    """Evaluate trained model"""
    logger.info("="*60)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("="*60)
    
    evaluator = ModelEvaluator(config)
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    test_yaml = Path(args.dataset_path) / 'dataset.yaml'
    
    if not test_yaml.exists():
        logger.error(f"Dataset YAML not found: {test_yaml}")
        sys.exit(1)
    
    metrics = evaluator.evaluate_model(
        model_path=args.model_path,
        test_data_yaml=str(test_yaml),
        save_visualizations=True
    )
    
    # Check performance thresholds
    logger.info("\nChecking performance thresholds...")
    thresholds = config.PERFORMANCE_THRESHOLDS
    
    passed = True
    if metrics['precision'] < thresholds['min_precision']:
        logger.warning(f"Precision below threshold: {metrics['precision']:.4f} < {thresholds['min_precision']}")
        passed = False
    
    if metrics['recall'] < thresholds['min_recall']:
        logger.warning(f"Recall below threshold: {metrics['recall']:.4f} < {thresholds['min_recall']}")
        passed = False
    
    if metrics['map50'] < thresholds['min_map50']:
        logger.warning(f"mAP@0.5 below threshold: {metrics['map50']:.4f} < {thresholds['min_map50']}")
        passed = False
    
    if passed:
        logger.info("✓ All performance thresholds met!")
    else:
        logger.warning("⚠ Some performance thresholds not met")
    
    # Measure inference time
    if args.measure_inference:
        logger.info("\nMeasuring inference time on target hardware...")
        
        # Get test images
        test_images_dir = Path(args.dataset_path) / 'test' / 'images'
        test_images = list(test_images_dir.glob('*.jpg'))[:10]  # Use first 10 images
        
        if len(test_images) > 0:
            timing_stats = evaluator.evaluate_inference_time(
                model_path=args.model_path,
                test_images=[str(img) for img in test_images],
                num_iterations=args.inference_iterations
            )
            
            logger.info(f"Average inference time: {timing_stats['mean_inference_time']:.4f}s")
            logger.info(f"FPS: {timing_stats['fps']:.2f}")
        else:
            logger.warning("No test images found for inference timing")
    
    logger.info("Evaluation completed ✓")
    
    return metrics


def compare_models(args, logger):
    """Compare multiple model architectures"""
    logger.info("="*60)
    logger.info("STEP 6: MODEL COMPARISON")
    logger.info("="*60)
    
    evaluator = ModelEvaluator(config)
    
    # Train and evaluate alternative models
    trainer = ModelTrainer(config)
    dataset_yaml = Path(args.dataset_path) / 'dataset.yaml'
    
    model_results = []
    
    # Evaluate base model
    logger.info(f"Evaluating base model: {args.model_path}")
    base_metrics = evaluator.evaluate_model(
        model_path=args.model_path,
        test_data_yaml=str(dataset_yaml),
        save_visualizations=False
    )
    model_results.append(base_metrics)
    
    # Train and evaluate alternative models
    for model_type in args.alternative_models:
        logger.info(f"\nTraining alternative model: {model_type}")
        
        try:
            # Train
            train_results = trainer.train_alternative_model(
                model_type=model_type,
                data_yaml=str(dataset_yaml)
            )
            
            # Evaluate
            eval_results = evaluator.evaluate_model(
                model_path=train_results['best_model_path'],
                test_data_yaml=str(dataset_yaml),
                save_visualizations=False
            )
            
            model_results.append(eval_results)
            
        except Exception as e:
            logger.error(f"Error with {model_type}: {e}")
    
    # Compare results
    comparison = evaluator.compare_models(model_results)
    
    logger.info("Model comparison completed ✓")
    
    return comparison


def process_orders(args, logger):
    """Process production orders"""
    logger.info("="*60)
    logger.info("STEP 7: ORDER PROCESSING")
    logger.info("="*60)
    
    # Initialize processor
    processor = OrderProcessor(config)
    
    # Optimize for Raspberry Pi if needed
    if args.optimize_rpi:
        logger.info("Optimizing model for Raspberry Pi 4B...")
        optimizer = RaspberryPiOptimizer(config)
        
        optimized_model = optimizer.optimize_model(args.model_path)
        optimizer.set_cpu_threads()
        optimizer.enable_memory_optimization()
        
        processor.load_model(optimized_model)
    else:
        processor.load_model(args.model_path)
    
    # Process orders
    if args.order_json:
        # Single order from JSON
        logger.info(f"Processing order from JSON: {args.order_json}")
        result = processor.process_order_from_json(
            order_json_path=args.order_json,
            real_photo_path=args.photo_path
        )
        
        logger.info(f"\nResult: {result.decision}")
        logger.info(f"Confidence: {result.confidence:.2f}")
        logger.info(f"Processing time: {result.processing_time:.3f}s")
        
    elif args.batch_process:
        # Batch processing
        logger.info("Processing batch of orders...")
        
        # Load orders from directory
        orders_dir = Path(args.orders_dir)
        photos_dir = Path(args.photos_dir)
        
        order_files = list(orders_dir.glob('*_order.json'))
        
        order_list = []
        for order_file in order_files:
            with open(order_file, 'r') as f:
                import json
                order_data = json.load(f)
            
            order = OrderRequest(**order_data)
            photo_path = photos_dir / f"{order.order_id}.jpg"
            
            if photo_path.exists():
                order_list.append((order, str(photo_path)))
        
        results = processor.batch_process_orders(order_list)
        
        logger.info(f"\nProcessed {len(results)} orders")
        logger.info(f"Correct: {sum(1 for r in results if r.decision == 'RIGHT')}")
        logger.info(f"Incorrect: {sum(1 for r in results if r.decision == 'WRONG')}")
    
    else:
        # Create and process single order
        logger.info("Creating new order...")
        
        order = processor.create_order_request(
            order_id=args.order_id,
            model_type=args.model_type,
            reference_image=args.reference_image
        )
        
        result = processor.process_order(order, args.photo_path)
        
        logger.info(f"\nResult: {result.decision}")
        logger.info(f"Confidence: {result.confidence:.2f}")
        logger.info(f"Processing time: {result.processing_time:.3f}s")
    
    # Print statistics
    stats = processor.get_statistics()
    logger.info("\nProcessing Statistics:")
    logger.info(f"Total orders: {stats['total_orders']}")
    logger.info(f"Accuracy: {stats['accuracy']:.2%}")
    logger.info(f"Avg processing time: {stats['avg_processing_time']:.3f}s")
    
    logger.info("Order processing completed ✓")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='LEGO Assembly Error Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Prepare dataset
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset')
    prepare_parser.add_argument('--images-path', required=True, help='Path to images directory')
    prepare_parser.add_argument('--labels-path', required=True, help='Path to labels directory')
    prepare_parser.add_argument('--output-path', default='data/prepared_dataset', help='Output directory')
    prepare_parser.add_argument('--kfold', action='store_true', help='Create K-fold splits')
    prepare_parser.add_argument('--kfold-output', default='data/kfold_dataset', help='K-fold output directory')
    
    # Train model
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--dataset-path', required=True, help='Path to prepared dataset')
    train_parser.add_argument('--model-name', default='best_model.pt', help='Output model name')
    train_parser.add_argument('--use-kfold', action='store_true', help='Use K-fold cross-validation')
    train_parser.add_argument('--kfold-output', default='data/kfold_dataset', help='K-fold dataset directory')
    
    # Fine-tune
    finetune_parser = subparsers.add_parser('finetune', help='Few-shot fine-tuning')
    finetune_parser.add_argument('--model-path', required=True, help='Path to base model')
    finetune_parser.add_argument('--real-photos-path', required=True, help='Path to real photos')
    finetune_parser.add_argument('--real-labels-path', required=True, help='Path to real photo labels')
    
    # Evaluate model
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model-path', required=True, help='Path to trained model')
    eval_parser.add_argument('--dataset-path', required=True, help='Path to dataset with test split')
    eval_parser.add_argument('--measure-inference', action='store_true', help='Measure inference time')
    eval_parser.add_argument('--inference-iterations', type=int, default=100, help='Inference iterations')
    
    # Compare models
    compare_parser = subparsers.add_parser('compare', help='Compare model architectures')
    compare_parser.add_argument('--model-path', required=True, help='Path to base model')
    compare_parser.add_argument('--dataset-path', required=True, help='Path to dataset')
    compare_parser.add_argument('--alternative-models', nargs='+', 
                               choices=list(config.ALTERNATIVE_MODELS.keys()),
                               help='Alternative models to compare')
    
    # Process orders
    process_parser = subparsers.add_parser('process', help='Process orders')
    process_parser.add_argument('--model-path', required=True, help='Path to trained model')
    process_parser.add_argument('--optimize-rpi', action='store_true', help='Optimize for Raspberry Pi')
    
    # Single order
    process_parser.add_argument('--order-json', help='Path to order JSON file')
    process_parser.add_argument('--photo-path', help='Path to captured photo')
    
    # Or create new order
    process_parser.add_argument('--order-id', help='Order ID')
    process_parser.add_argument('--model-type', help='LEGO model type')
    process_parser.add_argument('--reference-image', help='Reference image path')
    
    # Or batch process
    process_parser.add_argument('--batch-process', action='store_true', help='Batch process orders')
    process_parser.add_argument('--orders-dir', help='Directory with order JSON files')
    process_parser.add_argument('--photos-dir', help='Directory with photos')
    
    # Full pipeline
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--images-path', required=True, help='Path to images')
    pipeline_parser.add_argument('--labels-path', required=True, help='Path to labels')
    pipeline_parser.add_argument('--use-kfold', action='store_true', help='Use K-fold')
    pipeline_parser.add_argument('--real-photos-path', help='Path to real photos for fine-tuning')
    pipeline_parser.add_argument('--real-labels-path', help='Path to real photo labels')
    pipeline_parser.add_argument('--skip-training', action='store_true', help='Skip training if model exists')
    pipeline_parser.add_argument('--model-path', help='Existing model path (if skip-training)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(config.LOGS_DIR)
    
    logger.info("="*60)
    logger.info("LEGO ASSEMBLY ERROR DETECTION SYSTEM")
    logger.info("="*60)
    logger.info(f"Command: {args.command}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)
    
    try:
        if args.command == 'prepare':
            dataset_info = prepare_dataset(args, logger)
            if args.kfold:
                fold_info = prepare_kfold_dataset(args, logger)
        
        elif args.command == 'train':
            best_model = train_model(args, logger)
        
        elif args.command == 'finetune':
            finetuned_model = fine_tune_few_shot(args, logger)
        
        elif args.command == 'evaluate':
            metrics = evaluate_model(args, logger)
        
        elif args.command == 'compare':
            comparison = compare_models(args, logger)
        
        elif args.command == 'process':
            process_orders(args, logger)
        
        elif args.command == 'pipeline':
            # Run complete pipeline
            logger.info("Running complete pipeline...")
            
            # 1. Prepare dataset
            dataset_info = prepare_dataset(args, logger)
            
            if args.use_kfold:
                fold_info = prepare_kfold_dataset(args, logger)
                args.kfold_output = 'data/kfold_dataset'
            
            # 2. Train model (if not skipped)
            if not args.skip_training:
                args.dataset_path = dataset_info['output_path']
                args.model_name = 'best_model.pt'
                best_model = train_model(args, logger)
            else:
                best_model = args.model_path
                logger.info(f"Using existing model: {best_model}")
            
            # 3. Few-shot fine-tuning (if real photos provided)
            if args.real_photos_path:
                args.model_path = best_model
                finetuned_model = fine_tune_few_shot(args, logger)
                best_model = finetuned_model
            
            # 4. Evaluate model
            args.model_path = best_model
            args.dataset_path = dataset_info['output_path']
            args.measure_inference = True
            args.inference_iterations = 100
            metrics = evaluate_model(args, logger)
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Final model: {best_model}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"mAP@0.5: {metrics['map50']:.4f}")
            logger.info("="*60)
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
