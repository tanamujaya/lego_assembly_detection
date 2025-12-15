"""
Inference System for LEGO Assembly Error Detection
Optimized for Raspberry Pi 4B deployment
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np

from config import Config
from model_manager import ModelManager


class LEGOAssemblyInspector:
    """
    Production inference system for LEGO assembly error detection
    Optimized for Raspberry Pi 4B
    """
    
    def __init__(self, model_path: str, config: Config = None, 
                 use_optimization: bool = True):
        """
        Initialize the inspector
        
        Args:
            model_path: Path to trained model weights
            config: Configuration object
            use_optimization: Apply Raspberry Pi optimizations
        """
        self.config = config if config else Config()
        self.model_path = model_path
        self.use_optimization = use_optimization
        
        # Determine model type from path
        self.model_type = self._infer_model_type(model_path)
        
        # Initialize model
        self.model_manager = ModelManager(self.config)
        self.model = self.model_manager.get_model(self.model_type)
        
        # Load model with optimizations
        self._load_optimized_model()
        
        # Statistics tracking
        self.stats = {
            'total_inspections': 0,
            'total_errors_detected': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }
        
    def _infer_model_type(self, model_path: str) -> str:
        """Infer model type from model path"""
        for model_type in self.model_manager.list_models():
            if model_type in model_path:
                return model_type
        return self.config.DEFAULT_MODEL
    
    def _load_optimized_model(self):
        """Load model with Raspberry Pi optimizations"""
        print(f"Loading model: {self.model_path}")
        print(f"Model type: {self.model_type}")
        
        if self.use_optimization:
            print("Applying Raspberry Pi optimizations...")
            
            # Set number of threads for CPU inference
            import os
            os.environ['OMP_NUM_THREADS'] = str(
                self.config.RPI_OPTIMIZATIONS['num_threads']
            )
        
        self.model.load_model(self.model_path)
        
        print("Model loaded successfully!")
        
        # Warm-up inference
        print("Performing warm-up inference...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_path = '/tmp/dummy.jpg'
        cv2.imwrite(dummy_path, dummy_img)
        self.model.predict(dummy_path)
        print("Warm-up complete!")
    
    def inspect_assembly(self, order_request: Dict, 
                        image_path: str,
                        save_result: bool = True) -> Dict:
        """
        Inspect a LEGO assembly for errors
        
        Args:
            order_request: Dictionary containing order information
                          (e.g., {'model_id': '12345', 'model_name': 'Castle'})
            image_path: Path to the captured image of the assembly
            save_result: Whether to save inspection results
            
        Returns:
            Dictionary with inspection results
        """
        inspection_id = f"INS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Inspection ID: {inspection_id}")
        print(f"Order: {order_request.get('model_name', 'Unknown')}")
        print(f"{'='*60}\n")
        
        # Validate image exists
        if not Path(image_path).exists():
            return {
                'inspection_id': inspection_id,
                'status': 'error',
                'message': f"Image not found: {image_path}",
                'result': 'Unknown'
            }
        
        # Perform inference
        start_time = time.time()
        prediction = self.model.predict(
            image_path,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD
        )
        inference_time = time.time() - start_time
        
        # Determine result
        has_errors = prediction['has_errors']
        result = 'Wrong' if has_errors else 'Right'
        
        # Count errors
        error_count = sum(
            1 for det in prediction['detections'] 
            if det['class_name'] == 'error'
        )
        
        # Update statistics
        self.stats['total_inspections'] += 1
        if has_errors:
            self.stats['total_errors_detected'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['average_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['total_inspections']
        )
        
        # Compile inspection report
        inspection_result = {
            'inspection_id': inspection_id,
            'timestamp': datetime.now().isoformat(),
            'order_request': order_request,
            'image_path': image_path,
            'result': result,
            'has_errors': has_errors,
            'error_count': error_count,
            'confidence_scores': [
                det['confidence'] for det in prediction['detections']
            ],
            'detections': prediction['detections'],
            'assessment_time': inference_time,
            'model_used': self.model_type,
            'model_path': self.model_path
        }
        
        # Display results
        print(f"Result: {result}")
        print(f"Errors detected: {error_count}")
        print(f"Assessment time: {inference_time:.3f} seconds")
        print(f"{'='*60}\n")
        
        # Save results if requested
        if save_result:
            self._save_inspection_result(inspection_result)
        
        return inspection_result
    
    def batch_inspect(self, inspections: list) -> list:
        """
        Perform batch inspection
        
        Args:
            inspections: List of (order_request, image_path) tuples
            
        Returns:
            List of inspection results
        """
        print(f"\n{'='*60}")
        print(f"Starting Batch Inspection: {len(inspections)} items")
        print(f"{'='*60}\n")
        
        results = []
        start_time = time.time()
        
        for i, (order_request, image_path) in enumerate(inspections, 1):
            print(f"Processing {i}/{len(inspections)}...")
            result = self.inspect_assembly(order_request, image_path, save_result=False)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Batch summary
        correct_count = sum(1 for r in results if r['result'] == 'Right')
        error_count = sum(1 for r in results if r['result'] == 'Wrong')
        
        print(f"\n{'='*60}")
        print(f"Batch Inspection Complete!")
        print(f"{'='*60}")
        print(f"Total items: {len(inspections)}")
        print(f"Correct assemblies: {correct_count}")
        print(f"Assemblies with errors: {error_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per item: {total_time/len(inspections):.3f} seconds")
        print(f"{'='*60}\n")
        
        # Save batch results
        batch_result = {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(inspections),
            'correct_count': correct_count,
            'error_count': error_count,
            'total_time': total_time,
            'average_time': total_time / len(inspections),
            'results': results
        }
        
        batch_path = self.config.RESULTS_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_path, 'w') as f:
            json.dump(batch_result, f, indent=2)
        
        print(f"Batch results saved to: {batch_path}")
        
        return results
    
    def _save_inspection_result(self, result: Dict):
        """Save inspection result to JSON file"""
        inspection_dir = self.config.RESULTS_DIR / 'inspections'
        inspection_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result['inspection_id']}.json"
        filepath = inspection_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get inspection statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset inspection statistics"""
        self.stats = {
            'total_inspections': 0,
            'total_errors_detected': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }
    
    def visualize_result(self, inspection_result: Dict, 
                        output_path: str = None) -> str:
        """
        Create visualization of inspection result
        
        Args:
            inspection_result: Result from inspect_assembly
            output_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        import cv2
        
        # Load image
        img_path = inspection_result['image_path']
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not load image: {img_path}")
            return None
        
        # Draw detections
        for detection in inspection_result['detections']:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color: Red for errors, Green for correct
            color = (0, 0, 255) if class_name == 'error' else (0, 255, 0)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add result text
        result_text = f"Result: {inspection_result['result']}"
        time_text = f"Time: {inspection_result['assessment_time']:.3f}s"
        
        cv2.putText(img, result_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if inspection_result['result'] == 'Right' else (0, 0, 255), 2)
        cv2.putText(img, time_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        if output_path is None:
            vis_dir = self.config.RESULTS_DIR / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(vis_dir / f"{inspection_result['inspection_id']}.jpg")
        
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to: {output_path}")
        
        return output_path


class ProductionSimulator:
    """Simulate production environment for testing"""
    
    def __init__(self, inspector: LEGOAssemblyInspector):
        self.inspector = inspector
        
    def simulate_production_line(self, test_images_dir: Path, 
                                num_orders: int = None) -> Dict:
        """
        Simulate production line with multiple orders
        
        Args:
            test_images_dir: Directory containing test images
            num_orders: Number of orders to process (None = all)
            
        Returns:
            Summary of production run
        """
        print(f"\n{'='*60}")
        print(f"Starting Production Line Simulation")
        print(f"{'='*60}\n")
        
        # Collect test images
        image_files = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
        
        if num_orders:
            image_files = image_files[:num_orders]
        
        # Create orders
        orders = []
        for i, img_path in enumerate(image_files, 1):
            order = {
                'order_id': f"ORD_{i:04d}",
                'model_name': img_path.stem,
                'model_id': f"MDL_{i:03d}"
            }
            orders.append((order, str(img_path)))
        
        # Process orders
        results = self.inspector.batch_inspect(orders)
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r['result'] == 'Right')
        errors = sum(1 for r in results if r['result'] == 'Wrong')
        accuracy = correct / total if total > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_orders': total,
            'correct_assemblies': correct,
            'error_assemblies': errors,
            'accuracy': accuracy,
            'statistics': self.inspector.get_statistics()
        }
        
        print(f"\nProduction Summary:")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Average assessment time: {summary['statistics']['average_inference_time']:.3f}s")
        
        return summary
