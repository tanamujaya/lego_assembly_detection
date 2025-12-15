"""
Multi-View Deployment System for LEGO Assembly Error Detection
Handles 4-view capture, inference, and decision aggregation
"""

import cv2
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class MultiViewInspector:
    """
    4-View LEGO Assembly Inspector
    
    Captures images from 4 angles (0°, 90°, 180°, 270°), runs inference on each,
    and makes a final decision using multi-view aggregation logic.
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.5,
                 decision_strategy: str = 'any_error',
                 save_results: bool = True,
                 output_dir: str = './inspection_results'):
        """
        Initialize Multi-View Inspector
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detection
            decision_strategy: How to aggregate views ('any_error', 'majority_vote', 'all_error')
            save_results: Whether to save inspection results
            output_dir: Directory to save results
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.decision_strategy = decision_strategy
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.view_angles = [0, 90, 180, 270]
        self.class_names = {0: 'correct', 1: 'error'}
        
        print(f"Multi-View Inspector initialized")
        print(f"Model: {model_path}")
        print(f"Decision strategy: {decision_strategy}")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def inspect_assembly_multiview(self,
                                   image_paths: Dict[int, str],
                                   model_id: str = None) -> Dict:
        """
        Perform 4-view inspection on a LEGO model
        
        Args:
            image_paths: Dictionary mapping view angles to image paths
                        Example: {0: 'path/to/0deg.png', 90: 'path/to/90deg.png', ...}
            model_id: Optional model identifier for tracking
        
        Returns:
            Dictionary with inspection results
        """
        if model_id is None:
            model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        inspection_id = f"INS_{model_id}"
        timestamp = datetime.now().isoformat()
        
        print(f"\n{'='*70}")
        print(f"Starting 4-View Inspection: {inspection_id}")
        print(f"{'='*70}")
        
        # Validate input
        if not all(angle in image_paths for angle in self.view_angles):
            missing = [a for a in self.view_angles if a not in image_paths]
            raise ValueError(f"Missing views: {missing}")
        
        # Process each view
        view_results = []
        view_images = {}
        
        for angle in self.view_angles:
            image_path = image_paths[angle]
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ERROR: Could not load image for view {angle}°: {image_path}")
                continue
            
            view_images[angle] = image
            
            # Run inference
            view_result = self._inference_single_view(image, angle)
            view_results.append(view_result)
            
            # Print result
            status = "✓ CORRECT" if view_result['predicted_class'] == 0 else "✗ ERROR"
            print(f"View {angle:3d}°: {status} (confidence: {view_result['confidence']:.3f})")
        
        # Aggregate results across all views
        final_decision = self._aggregate_multiview_decision(view_results)
        
        # Prepare detailed result
        result = {
            'inspection_id': inspection_id,
            'model_id': model_id,
            'timestamp': timestamp,
            'final_decision': final_decision['decision'],
            'final_decision_label': 'CORRECT' if final_decision['decision'] == 0 else 'INCORRECT',
            'decision_strategy': self.decision_strategy,
            'confidence': final_decision['confidence'],
            'view_results': view_results,
            'view_agreement': final_decision['view_agreement'],
            'summary': {
                'total_views': len(view_results),
                'correct_views': sum(1 for v in view_results if v['predicted_class'] == 0),
                'error_views': sum(1 for v in view_results if v['predicted_class'] == 1),
                'avg_confidence': np.mean([v['confidence'] for v in view_results])
            }
        }
        
        # Print final decision
        print(f"{'='*70}")
        decision_label = result['final_decision_label']
        decision_emoji = "✅" if decision_label == "CORRECT" else "❌"
        print(f"{decision_emoji} FINAL DECISION: {decision_label}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   View breakdown: {result['summary']['correct_views']} correct, "
              f"{result['summary']['error_views']} error")
        print(f"{'='*70}\n")
        
        # Save results if requested
        if self.save_results:
            self._save_inspection_results(result, view_images, inspection_id)
        
        return result
    
    def _inference_single_view(self, image: np.ndarray, angle: int) -> Dict:
        """
        Run inference on a single view
        
        Args:
            image: Input image (BGR format from cv2)
            angle: View angle (0, 90, 180, 270)
        
        Returns:
            Dictionary with prediction results for this view
        """
        # Run YOLO inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        # Extract predictions
        if len(results[0].boxes) == 0:
            # No detections - assume correct (or could assume error, depending on use case)
            return {
                'view_angle': angle,
                'predicted_class': 0,  # Default to correct if no detection
                'class_name': 'correct',
                'confidence': 0.0,
                'detected': False,
                'num_detections': 0
            }
        
        # Get highest confidence detection
        boxes = results[0].boxes
        max_conf_idx = boxes.conf.argmax()
        
        class_id = int(boxes.cls[max_conf_idx])
        confidence = float(boxes.conf[max_conf_idx])
        class_name = self.class_names.get(class_id, f'class_{class_id}')
        
        # Get bounding box
        bbox = boxes.xyxy[max_conf_idx].cpu().numpy()
        
        return {
            'view_angle': angle,
            'predicted_class': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'detected': True,
            'num_detections': len(boxes),
            'bbox': bbox.tolist()
        }
    
    def _aggregate_multiview_decision(self, view_results: List[Dict]) -> Dict:
        """
        Aggregate predictions from all views into final decision
        
        Args:
            view_results: List of prediction results from each view
        
        Returns:
            Dictionary with aggregated decision
        """
        predictions = [v['predicted_class'] for v in view_results]
        confidences = [v['confidence'] for v in view_results]
        
        # Calculate view agreement
        view_agreement = len(set(predictions)) == 1
        
        # Apply decision strategy
        if self.decision_strategy == 'any_error':
            # If ANY view shows error → final decision = ERROR
            final_class = 1 if 1 in predictions else 0
            
        elif self.decision_strategy == 'majority_vote':
            # Use majority vote
            error_count = sum(1 for p in predictions if p == 1)
            final_class = 1 if error_count > len(predictions) / 2 else 0
            
        elif self.decision_strategy == 'all_error':
            # If ALL views show error → final decision = ERROR
            final_class = 1 if all(p == 1 for p in predictions) else 0
            
        else:
            raise ValueError(f"Unknown decision strategy: {self.decision_strategy}")
        
        # Aggregate confidence
        if final_class == 1:
            # Use max confidence of error predictions
            error_confidences = [c for p, c in zip(predictions, confidences) if p == 1]
            final_confidence = max(error_confidences) if error_confidences else 0.0
        else:
            # Use min confidence of correct predictions
            correct_confidences = [c for p, c in zip(predictions, confidences) if p == 0]
            final_confidence = min(correct_confidences) if correct_confidences else 0.0
        
        return {
            'decision': final_class,
            'confidence': final_confidence,
            'view_agreement': view_agreement,
            'predictions_per_view': predictions
        }
    
    def _save_inspection_results(self, 
                                 result: Dict,
                                 view_images: Dict[int, np.ndarray],
                                 inspection_id: str):
        """
        Save inspection results and annotated images
        
        Args:
            result: Inspection result dictionary
            view_images: Dictionary of view images
            inspection_id: Unique inspection identifier
        """
        # Create output directory for this inspection
        inspection_dir = self.output_dir / inspection_id
        inspection_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON result
        result_path = inspection_dir / 'result.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save individual view images with annotations
        for view_result in result['view_results']:
            angle = view_result['view_angle']
            image = view_images.get(angle)
            
            if image is None:
                continue
            
            # Annotate image
            annotated = image.copy()
            
            # Add view angle label
            cv2.putText(annotated, f"{angle} deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add prediction label
            label = view_result['class_name'].upper()
            conf = view_result['confidence']
            color = (0, 255, 0) if view_result['predicted_class'] == 0 else (0, 0, 255)
            
            cv2.putText(annotated, f"{label} ({conf:.2f})", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw bounding box if detected
            if view_result['detected'] and 'bbox' in view_result:
                bbox = view_result['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Save annotated image
            image_path = inspection_dir / f"view_{angle}.png"
            cv2.imwrite(str(image_path), annotated)
        
        # Create 4-panel summary image
        self._create_summary_image(view_images, result, inspection_dir)
        
        print(f"Results saved to: {inspection_dir}")
    
    def _create_summary_image(self,
                             view_images: Dict[int, np.ndarray],
                             result: Dict,
                             output_dir: Path):
        """Create a 4-panel summary image showing all views"""
        if len(view_images) != 4:
            return
        
        # Get image size
        h, w = list(view_images.values())[0].shape[:2]
        
        # Create 2x2 grid
        summary = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Place images in grid
        positions = {
            0: (0, 0),      # Top-left
            90: (0, 1),     # Top-right
            180: (1, 0),    # Bottom-left
            270: (1, 1)     # Bottom-right
        }
        
        for angle, (row, col) in positions.items():
            if angle in view_images:
                y_start = row * h
                x_start = col * w
                summary[y_start:y_start+h, x_start:x_start+w] = view_images[angle]
                
                # Add angle label
                cv2.putText(summary, f"{angle} deg", 
                           (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add final decision overlay
        decision = result['final_decision_label']
        color = (0, 255, 0) if decision == "CORRECT" else (0, 0, 255)
        
        # Add semi-transparent overlay at bottom
        overlay = summary.copy()
        cv2.rectangle(overlay, (0, h * 2 - 80), (w * 2, h * 2), (0, 0, 0), -1)
        summary = cv2.addWeighted(summary, 0.7, overlay, 0.3, 0)
        
        # Add text
        text = f"FINAL: {decision} ({result['confidence']:.2f})"
        cv2.putText(summary, text, (w - 200, h * 2 - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Save summary
        summary_path = output_dir / 'summary_4view.png'
        cv2.imwrite(str(summary_path), summary)


class MultiViewBatchEvaluator:
    """
    Evaluate multi-view system on test dataset
    Groups images by model and evaluates model-level accuracy
    """
    
    def __init__(self,
                 model_path: str,
                 confidence_threshold: float = 0.5,
                 decision_strategy: str = 'any_error'):
        """
        Initialize batch evaluator
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Confidence threshold for detections
            decision_strategy: Multi-view decision strategy
        """
        self.inspector = MultiViewInspector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            decision_strategy=decision_strategy,
            save_results=False
        )
    
    def evaluate_test_set(self,
                         images_dir: str,
                         labels_dir: str,
                         output_path: str = None) -> Dict:
        """
        Evaluate multi-view system on test set
        
        Args:
            images_dir: Directory containing test images
            labels_dir: Directory containing ground truth labels
            output_path: Optional path to save results
        
        Returns:
            Dictionary with evaluation metrics
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        # Group images by model
        model_groups = self._group_images_by_model(images_dir)
        
        print(f"\n{'='*70}")
        print(f"Multi-View Batch Evaluation")
        print(f"{'='*70}")
        print(f"Total models to evaluate: {len(model_groups)}")
        print(f"Decision strategy: {self.inspector.decision_strategy}")
        print(f"{'='*70}\n")
        
        results = {
            'total_models': len(model_groups),
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'per_model_results': [],
            'per_view_metrics': defaultdict(lambda: {'correct': 0, 'total': 0})
        }
        
        # Evaluate each model
        for model_id, image_paths in model_groups.items():
            # Get ground truth
            ground_truth = self._get_ground_truth(image_paths, labels_dir)
            
            # Run inspection
            try:
                inspection_result = self.inspector.inspect_assembly_multiview(
                    image_paths=image_paths,
                    model_id=str(model_id)
                )
                
                predicted = inspection_result['final_decision']
                is_correct = (predicted == ground_truth)
                
                if is_correct:
                    results['correct_predictions'] += 1
                else:
                    results['incorrect_predictions'] += 1
                
                # Store per-model result
                results['per_model_results'].append({
                    'model_id': model_id,
                    'ground_truth': ground_truth,
                    'predicted': predicted,
                    'correct': is_correct,
                    'view_results': inspection_result['view_results']
                })
                
                # Update per-view metrics
                for view_result in inspection_result['view_results']:
                    angle = view_result['view_angle']
                    view_pred = view_result['predicted_class']
                    results['per_view_metrics'][angle]['total'] += 1
                    if view_pred == ground_truth:
                        results['per_view_metrics'][angle]['correct'] += 1
                
                # Progress
                if len(results['per_model_results']) % 10 == 0:
                    print(f"  Progress: {len(results['per_model_results'])}/{len(model_groups)}")
                    
            except Exception as e:
                print(f"  ERROR evaluating model {model_id}: {e}")
        
        # Calculate metrics
        accuracy = results['correct_predictions'] / results['total_models']
        
        # Per-view accuracy
        per_view_accuracy = {}
        for angle, metrics in results['per_view_metrics'].items():
            per_view_accuracy[angle] = metrics['correct'] / metrics['total']
        
        results['model_accuracy'] = accuracy
        results['per_view_accuracy'] = per_view_accuracy
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Model-level Accuracy: {accuracy:.2%}")
        print(f"Correct: {results['correct_predictions']}/{results['total_models']}")
        print(f"\nPer-View Accuracy:")
        for angle in sorted(per_view_accuracy.keys()):
            print(f"  {angle:3d}°: {per_view_accuracy[angle]:.2%}")
        print(f"{'='*70}\n")
        
        # Save results if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                # Convert defaultdict to dict for JSON serialization
                results_copy = dict(results)
                results_copy['per_view_metrics'] = dict(results['per_view_metrics'])
                json.dump(results_copy, f, indent=2)
            print(f"Results saved to: {output_path}")
        
        return results
    
    def _group_images_by_model(self, images_dir: Path) -> Dict[int, Dict[int, str]]:
        """Group images by model ID based on filename pattern"""
        from multiview_config import get_model_id_from_filename, get_view_angle_from_filename
        
        model_groups = defaultdict(dict)
        
        for img_path in images_dir.glob('*.png'):
            try:
                model_id = get_model_id_from_filename(img_path.name)
                angle = get_view_angle_from_filename(img_path.name)
                model_groups[model_id][angle] = str(img_path)
            except Exception as e:
                print(f"Warning: Could not parse {img_path.name}: {e}")
        
        # Filter to only complete models (all 4 views)
        complete_models = {
            mid: views for mid, views in model_groups.items()
            if len(views) == 4 and all(a in views for a in [0, 90, 180, 270])
        }
        
        return complete_models
    
    def _get_ground_truth(self, 
                         image_paths: Dict[int, str],
                         labels_dir: Path) -> int:
        """Get ground truth label (should be same for all views of a model)"""
        # Use first view's label
        first_image = Path(list(image_paths.values())[0])
        label_path = labels_dir / f"{first_image.stem}.txt"
        
        if not label_path.exists():
            raise ValueError(f"Label not found: {label_path}")
        
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
            if label_content:
                class_id = int(label_content.split()[0])
                return class_id
        
        return 0  # Default to correct


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_single_inspection():
    """Example: Inspect a single model with 4 views"""
    
    # Initialize inspector
    inspector = MultiViewInspector(
        model_path='path/to/your/best.pt',
        confidence_threshold=0.5,
        decision_strategy='any_error',
        save_results=True,
        output_dir='./multiview_results'
    )
    
    # Specify paths to 4 views of the same model
    image_paths = {
        0: 'path/to/image_0000_0.png',
        90: 'path/to/image_0000_90.png',
        180: 'path/to/image_0000_180.png',
        270: 'path/to/image_0000_270.png'
    }
    
    # Run inspection
    result = inspector.inspect_assembly_multiview(
        image_paths=image_paths,
        model_id='model_0000'
    )
    
    print(f"\nFinal Decision: {result['final_decision_label']}")
    print(f"Confidence: {result['confidence']:.3f}")


def example_batch_evaluation():
    """Example: Evaluate on test set"""
    
    evaluator = MultiViewBatchEvaluator(
        model_path='path/to/your/best.pt',
        confidence_threshold=0.5,
        decision_strategy='any_error'
    )
    
    results = evaluator.evaluate_test_set(
        images_dir='path/to/test/images',
        labels_dir='path/to/test/labels',
        output_path='./multiview_evaluation_results.json'
    )
    
    print(f"Model Accuracy: {results['model_accuracy']:.2%}")


if __name__ == "__main__":
    print("Multi-View Deployment System")
    print("See example_single_inspection() and example_batch_evaluation() for usage")
