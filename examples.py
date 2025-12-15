"""
Example Usage Scripts
Demonstrates various use cases of the LEGO Assembly Error Detection System
"""

from pathlib import Path
import json
from config import Config
from training_pipeline import TrainingPipeline
from inference import LEGOAssemblyInspector, ProductionSimulator
from evaluation import ModelEvaluator


# ============================================================================
# EXAMPLE 1: Complete Training Pipeline with K-Fold Cross-Validation
# ============================================================================

def example_1_complete_training():
    """Train a model with K-fold cross-validation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Complete Training Pipeline")
    print("="*70 + "\n")
    
    # Initialize
    config = Config()
    config.EPOCHS = 50  # Reduce for quick testing
    config.BATCH_SIZE = 16
    
    pipeline = TrainingPipeline(config)
    
    # Train with 10-fold cross-validation
    results = pipeline.train_kfold(
        model_type='yolov8n',
        dataset_dir=Path('./data/renders'),
        k=10
    )
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Mean mAP50: {results['aggregate_metrics']['mAP50_mean']:.4f}")
    print(f"Std mAP50: {results['aggregate_metrics']['mAP50_std']:.4f}")
    print(f"Best model: {results['best_model_path']}")
    
    return results


# ============================================================================
# EXAMPLE 2: Quick Training with Standard Split
# ============================================================================

def example_2_quick_training():
    """Quick training with standard 70/15/15 split"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Quick Training (Standard Split)")
    print("="*70 + "\n")
    
    # Initialize
    config = Config()
    config.EPOCHS = 30  # Quick training
    
    pipeline = TrainingPipeline(config)
    
    # Train with standard split
    results = pipeline.train_standard_split(
        model_type='yolov8n',
        dataset_dir=Path('./data/renders')
    )
    
    # Print results
    print("\nTest Set Performance:")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")
    print(f"mAP50: {results['test_metrics']['mAP50']:.4f}")
    
    return results


# ============================================================================
# EXAMPLE 3: Few-Shot Fine-Tuning on Real Photos
# ============================================================================

def example_3_few_shot_fine_tuning(base_model_path: str):
    """Fine-tune model on real photos"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Few-Shot Fine-Tuning")
    print("="*70 + "\n")
    
    # Initialize
    config = Config()
    config.FEW_SHOT_SAMPLES = 10  # Use 10 samples per class
    config.FINE_TUNE_EPOCHS = 50
    
    pipeline = TrainingPipeline(config)
    
    # Fine-tune on real photos
    results = pipeline.few_shot_fine_tune(
        base_model_path=base_model_path,
        few_shot_dir=Path('./data/real_photos')
    )
    
    print("\nFine-Tuning Results:")
    print(f"Validation mAP50: {results['val_metrics']['mAP50']:.4f}")
    print(f"Fine-tuned model: {results['fine_tune_results']['best_model_path']}")
    
    return results


# ============================================================================
# EXAMPLE 4: Single Image Inspection
# ============================================================================

def example_4_single_inspection(model_path: str, image_path: str):
    """Inspect a single LEGO assembly"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Single Image Inspection")
    print("="*70 + "\n")
    
    # Initialize inspector
    inspector = LEGOAssemblyInspector(
        model_path=model_path,
        use_optimization=True  # Enable RPi optimizations
    )
    
    # Create order request
    order_request = {
        'order_id': 'ORD_001',
        'model_name': 'Medieval Castle',
        'model_id': 'LEGO-31120',
        'customer': 'John Doe'
    }
    
    # Perform inspection
    result = inspector.inspect_assembly(
        order_request=order_request,
        image_path=image_path
    )
    
    # Display results
    print(f"\nInspection Results:")
    print(f"Order ID: {result['inspection_id']}")
    print(f"Result: {result['result']}")
    print(f"Assessment Time: {result['assessment_time']:.3f} seconds")
    
    if result['has_errors']:
        print(f"\n⚠️  {result['error_count']} error(s) detected!")
        for i, detection in enumerate(result['detections'], 1):
            if detection['class_name'] == 'error':
                print(f"  Error {i}: Confidence {detection['confidence']:.2%}")
    else:
        print("\n✅ Assembly is correct!")
    
    # Visualize result
    vis_path = inspector.visualize_result(result)
    print(f"\nVisualization saved: {vis_path}")
    
    return result


# ============================================================================
# EXAMPLE 5: Batch Processing
# ============================================================================

def example_5_batch_processing(model_path: str, images_dir: str):
    """Process multiple assemblies in batch"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Batch Processing")
    print("="*70 + "\n")
    
    # Initialize inspector
    inspector = LEGOAssemblyInspector(model_path=model_path)
    
    # Prepare batch of inspections
    images_path = Path(images_dir)
    inspections = []
    
    for i, img_file in enumerate(images_path.glob('*.jpg'), 1):
        order_request = {
            'order_id': f'ORD_{i:04d}',
            'model_name': img_file.stem,
            'model_id': f'MDL_{i:03d}'
        }
        inspections.append((order_request, str(img_file)))
    
    # Process batch
    results = inspector.batch_inspect(inspections)
    
    # Analyze results
    correct_count = sum(1 for r in results if r['result'] == 'Right')
    error_count = sum(1 for r in results if r['result'] == 'Wrong')
    
    print(f"\nBatch Processing Summary:")
    print(f"Total assemblies: {len(results)}")
    print(f"Correct: {correct_count} ({correct_count/len(results)*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/len(results)*100:.1f}%)")
    
    # Get statistics
    stats = inspector.get_statistics()
    print(f"\nPerformance:")
    print(f"Average assessment time: {stats['average_inference_time']:.3f} seconds")
    print(f"Throughput: {1/stats['average_inference_time']:.1f} assemblies/minute")
    
    return results


# ============================================================================
# EXAMPLE 6: Production Line Simulation
# ============================================================================

def example_6_production_simulation(model_path: str, test_dir: str):
    """Simulate a production line"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Production Line Simulation")
    print("="*70 + "\n")
    
    # Initialize
    inspector = LEGOAssemblyInspector(model_path=model_path)
    simulator = ProductionSimulator(inspector)
    
    # Run simulation
    summary = simulator.simulate_production_line(
        test_images_dir=Path(test_dir),
        num_orders=50  # Process 50 orders
    )
    
    print(f"\nProduction Line Summary:")
    print(f"Quality Rate: {summary['accuracy']*100:.1f}%")
    print(f"Throughput: {60/summary['statistics']['average_inference_time']:.1f} units/hour")
    
    return summary


# ============================================================================
# EXAMPLE 7: Model Comparison
# ============================================================================

def example_7_model_comparison():
    """Compare performance of multiple models"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Model Comparison")
    print("="*70 + "\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load results from different models
    results_files = [
        './results/yolov8n_kfold_results.json',
        './results/yolov8s_kfold_results.json',
        './results/yolov5n_kfold_results.json'
    ]
    
    results_list = []
    for results_file in results_files:
        if Path(results_file).exists():
            results = evaluator.load_results(Path(results_file))
            results_list.append(results)
    
    if len(results_list) < 2:
        print("Need at least 2 result files for comparison")
        return None
    
    # Create comparison visualizations
    comparison_plot = evaluator.plot_metrics_comparison(results_list)
    dashboard = evaluator.create_summary_dashboard(results_list)
    
    print(f"Comparison plot: {comparison_plot}")
    print(f"Dashboard: {dashboard}")
    
    return results_list


# ============================================================================
# EXAMPLE 8: Complete Workflow (Train → Fine-tune → Deploy)
# ============================================================================

def example_8_complete_workflow():
    """Complete workflow from training to deployment"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Complete Workflow")
    print("="*70 + "\n")
    
    config = Config()
    pipeline = TrainingPipeline(config)
    
    # Step 1: Train on rendered images
    print("\n[Step 1/4] Training on rendered images...")
    train_results = pipeline.train_kfold(
        model_type='yolov8n',
        dataset_dir=Path('./data/renders'),
        k=5  # Use 5 folds for quicker demo
    )
    
    best_model = train_results['best_model_path']
    print(f"✓ Training complete. Best model: {best_model}")
    
    # Step 2: Fine-tune on real photos
    print("\n[Step 2/4] Fine-tuning on real photos...")
    finetune_results = pipeline.few_shot_fine_tune(
        base_model_path=best_model,
        few_shot_dir=Path('./data/real_photos')
    )
    
    final_model = finetune_results['fine_tune_results']['best_model_path']
    print(f"✓ Fine-tuning complete. Final model: {final_model}")
    
    # Step 3: Evaluate on test set
    print("\n[Step 3/4] Evaluating final model...")
    evaluator = ModelEvaluator(config)
    report = evaluator.generate_evaluation_report(finetune_results)
    print(f"✓ Evaluation report: {report}")
    
    # Step 4: Deploy for production
    print("\n[Step 4/4] Running production test...")
    inspector = LEGOAssemblyInspector(model_path=final_model)
    
    # Test on a few images
    test_images = list(Path('./test_images').glob('*.jpg'))[:5]
    for img in test_images:
        order = {'model_name': img.stem}
        result = inspector.inspect_assembly(order, str(img))
        print(f"  {img.name}: {result['result']} ({result['assessment_time']:.3f}s)")
    
    print("\n✓ Complete workflow finished!")
    print(f"Final model ready for deployment: {final_model}")
    
    return final_model


# ============================================================================
# EXAMPLE 9: Custom Evaluation with Visualizations
# ============================================================================

def example_9_custom_evaluation(results_path: str):
    """Create custom evaluation visualizations"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Custom Evaluation")
    print("="*70 + "\n")
    
    evaluator = ModelEvaluator()
    
    # Load results
    results = evaluator.load_results(Path(results_path))
    
    # Generate comprehensive report
    report_path = evaluator.generate_evaluation_report(results)
    print(f"Report: {report_path}")
    
    # Create K-fold metrics plot
    if 'aggregate_metrics' in results:
        kfold_plot = evaluator.plot_kfold_metrics(results)
        print(f"K-fold plot: {kfold_plot}")
    
    # If we have inspection results, plot inference times
    inspection_results_dir = Path('./results/inspections')
    if inspection_results_dir.exists():
        inspection_results = []
        for json_file in inspection_results_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                inspection_results.append(json.load(f))
        
        if inspection_results:
            time_plot = evaluator.plot_inference_time_distribution(inspection_results)
            print(f"Inference time plot: {time_plot}")
    
    return results


# ============================================================================
# EXAMPLE 10: Raspberry Pi Deployment Test
# ============================================================================

def example_10_raspberry_pi_test(model_path: str):
    """Test model on Raspberry Pi with optimizations"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Raspberry Pi Deployment Test")
    print("="*70 + "\n")
    
    # Initialize with RPi optimizations
    config = Config()
    inspector = LEGOAssemblyInspector(
        model_path=model_path,
        config=config,
        use_optimization=True
    )
    
    # Run benchmark
    print("Running benchmark (10 images)...")
    test_images = list(Path('./test_images').glob('*.jpg'))[:10]
    
    times = []
    for img in test_images:
        order = {'model_name': img.stem}
        result = inspector.inspect_assembly(order, str(img), save_result=False)
        times.append(result['assessment_time'])
    
    # Statistics
    import numpy as np
    print(f"\nBenchmark Results:")
    print(f"Mean time: {np.mean(times):.3f}s")
    print(f"Std time: {np.std(times):.3f}s")
    print(f"Min time: {np.min(times):.3f}s")
    print(f"Max time: {np.max(times):.3f}s")
    print(f"Throughput: {1/np.mean(times):.1f} images/second")
    print(f"Expected hourly capacity: {int(3600/np.mean(times))} assemblies")
    
    return times


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Interactive menu for running examples"""
    print("\n" + "="*70)
    print("LEGO ASSEMBLY ERROR DETECTION - EXAMPLE SCRIPTS")
    print("="*70)
    print("\nAvailable Examples:")
    print("1.  Complete Training Pipeline (K-Fold)")
    print("2.  Quick Training (Standard Split)")
    print("3.  Few-Shot Fine-Tuning")
    print("4.  Single Image Inspection")
    print("5.  Batch Processing")
    print("6.  Production Line Simulation")
    print("7.  Model Comparison")
    print("8.  Complete Workflow (Train → Fine-tune → Deploy)")
    print("9.  Custom Evaluation")
    print("10. Raspberry Pi Deployment Test")
    print("\n0.  Exit")
    
    choice = input("\nSelect example (0-10): ").strip()
    
    if choice == '1':
        example_1_complete_training()
    elif choice == '2':
        example_2_quick_training()
    elif choice == '3':
        model_path = input("Enter base model path: ").strip()
        example_3_few_shot_fine_tuning(model_path)
    elif choice == '4':
        model_path = input("Enter model path: ").strip()
        image_path = input("Enter image path: ").strip()
        example_4_single_inspection(model_path, image_path)
    elif choice == '5':
        model_path = input("Enter model path: ").strip()
        images_dir = input("Enter images directory: ").strip()
        example_5_batch_processing(model_path, images_dir)
    elif choice == '6':
        model_path = input("Enter model path: ").strip()
        test_dir = input("Enter test directory: ").strip()
        example_6_production_simulation(model_path, test_dir)
    elif choice == '7':
        example_7_model_comparison()
    elif choice == '8':
        example_8_complete_workflow()
    elif choice == '9':
        results_path = input("Enter results JSON path: ").strip()
        example_9_custom_evaluation(results_path)
    elif choice == '10':
        model_path = input("Enter model path: ").strip()
        example_10_raspberry_pi_test(model_path)
    elif choice == '0':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    main()
