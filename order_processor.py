"""
Order Processing and Inference Module
Handles order requests, performs inference, and returns Right/Wrong decisions
"""

import os
import json
import time
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available")


@dataclass
class OrderRequest:
    """Data class for order request"""
    order_id: str
    model_type: str
    reference_image: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class InspectionResult:
    """Data class for inspection result"""
    order_id: str
    decision: str  # "RIGHT" or "WRONG"
    confidence: float
    detected_errors: List[str]
    processing_time: float
    timestamp: str
    details: Dict = None
    
    def to_dict(self):
        return asdict(self)


class OrderProcessor:
    """Processes orders and performs assembly error detection"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_path = None
        self.order_config = config.ORDER_CONFIG
        self.inference_config = config.INFERENCE_CONFIG
        
        # Create output directories
        self.order_config['order_input_path'].mkdir(parents=True, exist_ok=True)
        self.order_config['output_results_path'].mkdir(parents=True, exist_ok=True)
        self.inference_config['output_image_path'].mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_orders': 0,
            'correct_assemblies': 0,
            'incorrect_assemblies': 0,
            'avg_processing_time': 0,
            'processing_times': []
        }
    
    def load_model(self, model_path: str):
        """
        Load trained model for inference
        
        Args:
            model_path: Path to trained model file
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not installed")
        
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
        logger.info("Model loaded successfully")
    
    def process_order(self, 
                     order: OrderRequest,
                     real_photo_path: str) -> InspectionResult:
        """
        Process a single order and determine if assembly is correct
        
        Args:
            order: OrderRequest object
            real_photo_path: Path to captured photo of assembled model
            
        Returns:
            InspectionResult object
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")
        
        logger.info(f"Processing order: {order.order_id}")
        
        start_time = time.time()
        
        # Read image
        image = cv2.imread(real_photo_path)
        if image is None:
            raise ValueError(f"Could not read image: {real_photo_path}")
        
        # Run inference
        results = self.model(
            real_photo_path,
            conf=self.config.MODEL_CONFIG['confidence_threshold'],
            iou=self.config.MODEL_CONFIG['iou_threshold'],
            verbose=False
        )
        
        # Process results
        decision, confidence, errors, details = self._analyze_results(results[0])
        
        processing_time = time.time() - start_time
        
        # Create result
        result = InspectionResult(
            order_id=order.order_id,
            decision=decision,
            confidence=confidence,
            detected_errors=errors,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            details=details
        )
        
        # Update statistics
        self._update_stats(result)
        
        # Save annotated image if configured
        if self.inference_config['save_output_images']:
            self._save_annotated_image(image, results[0], result)
        
        # Save result
        self._save_result(result)
        
        logger.info(f"Order {order.order_id}: {decision} (confidence: {confidence:.2f}, "
                   f"time: {processing_time:.3f}s)")
        
        return result
    
    def process_order_from_json(self, order_json_path: str, real_photo_path: str) -> InspectionResult:
        """
        Process order from JSON file
        
        Args:
            order_json_path: Path to order JSON file
            real_photo_path: Path to real photo
            
        Returns:
            InspectionResult object
        """
        # Load order
        with open(order_json_path, 'r') as f:
            order_data = json.load(f)
        
        order = OrderRequest(**order_data)
        
        return self.process_order(order, real_photo_path)
    
    def batch_process_orders(self, 
                           order_list: List[Tuple[OrderRequest, str]]) -> List[InspectionResult]:
        """
        Process multiple orders in batch
        
        Args:
            order_list: List of (OrderRequest, photo_path) tuples
            
        Returns:
            List of InspectionResult objects
        """
        logger.info(f"Processing batch of {len(order_list)} orders")
        
        results = []
        for order, photo_path in order_list:
            try:
                result = self.process_order(order, photo_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing order {order.order_id}: {e}")
        
        # Generate batch report
        self._generate_batch_report(results)
        
        return results
    
    def _analyze_results(self, result) -> Tuple[str, float, List[str], Dict]:
        """
        Analyze detection results and determine RIGHT/WRONG
        
        Args:
            result: YOLO detection result
            
        Returns:
            Tuple of (decision, confidence, errors_list, details_dict)
        """
        detected_errors = []
        details = {
            'num_detections': len(result.boxes),
            'detections': []
        }
        
        error_confidence_scores = []
        correct_confidence_scores = []
        
        # Process each detection
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.config.CLASS_NAMES.get(class_id, 'unknown')
            
            detection_info = {
                'class': class_name,
                'confidence': confidence,
                'bbox': box.xyxy[0].tolist()
            }
            details['detections'].append(detection_info)
            
            # Check if it's an error
            if class_id == 1:  # assembly_error
                detected_errors.append(f"{class_name} (conf: {confidence:.2f})")
                error_confidence_scores.append(confidence)
            else:  # correct_assembly
                correct_confidence_scores.append(confidence)
        
        # Decision logic
        if len(detected_errors) > 0:
            decision = "WRONG"
            # Use highest error confidence as overall confidence
            confidence = max(error_confidence_scores)
        elif len(correct_confidence_scores) > 0:
            decision = "RIGHT"
            # Use average correct confidence
            confidence = np.mean(correct_confidence_scores)
        else:
            # No detections - inconclusive, default to WRONG for safety
            decision = "WRONG"
            confidence = 0.0
            detected_errors.append("No parts detected")
        
        return decision, confidence, detected_errors, details
    
    def _save_annotated_image(self, image: np.ndarray, result, inspection_result: InspectionResult):
        """Save image with detection annotations"""
        annotated = image.copy()
        
        # Draw bounding boxes
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Color based on class
            if class_id == 1:  # error
                color = self.inference_config['color_error']
                label = f"ERROR {confidence:.2f}"
            else:  # correct
                color = self.inference_config['color_correct']
                label = f"OK {confidence:.2f}"
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 
                         self.inference_config['line_thickness'])
            
            # Draw label
            if self.inference_config['show_labels']:
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 
                    self.inference_config['font_scale'], 2
                )
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          self.inference_config['font_scale'],
                          (255, 255, 255), 2)
        
        # Add overall decision
        decision_text = f"{inspection_result.decision}: {inspection_result.confidence:.2f}"
        decision_color = self.inference_config['color_correct'] if inspection_result.decision == "RIGHT" \
                        else self.inference_config['color_error']
        
        cv2.rectangle(annotated, (10, 10), (300, 60), decision_color, -1)
        cv2.putText(annotated, decision_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Save
        output_path = self.inference_config['output_image_path'] / \
                     f"{inspection_result.order_id}_result.jpg"
        cv2.imwrite(str(output_path), annotated)
        
        logger.info(f"Annotated image saved to {output_path}")
    
    def _save_result(self, result: InspectionResult):
        """Save inspection result to JSON"""
        result_path = self.order_config['output_results_path'] / \
                     f"{result.order_id}_result.json"
        
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _update_stats(self, result: InspectionResult):
        """Update processing statistics"""
        self.stats['total_orders'] += 1
        self.stats['processing_times'].append(result.processing_time)
        
        if result.decision == "RIGHT":
            self.stats['correct_assemblies'] += 1
        else:
            self.stats['incorrect_assemblies'] += 1
        
        self.stats['avg_processing_time'] = np.mean(self.stats['processing_times'])
    
    def _generate_batch_report(self, results: List[InspectionResult]):
        """Generate batch processing report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_orders': len(results),
            'correct': sum(1 for r in results if r.decision == "RIGHT"),
            'incorrect': sum(1 for r in results if r.decision == "WRONG"),
            'avg_processing_time': np.mean([r.processing_time for r in results]),
            'avg_confidence': np.mean([r.confidence for r in results]),
            'orders': [r.to_dict() for r in results]
        }
        
        report_path = self.order_config['output_results_path'] / \
                     f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Batch report saved to {report_path}")
    
    def get_statistics(self) -> Dict:
        """Get current processing statistics"""
        accuracy = self.stats['correct_assemblies'] / max(self.stats['total_orders'], 1)
        
        return {
            **self.stats,
            'accuracy': accuracy,
            'error_rate': 1 - accuracy
        }
    
    def create_order_request(self,
                           order_id: str,
                           model_type: str,
                           reference_image: Optional[str] = None) -> OrderRequest:
        """
        Create a new order request
        
        Args:
            order_id: Unique order identifier
            model_type: Type of LEGO model
            reference_image: Optional path to reference image
            
        Returns:
            OrderRequest object
        """
        order = OrderRequest(
            order_id=order_id,
            model_type=model_type,
            reference_image=reference_image
        )
        
        # Save order to file
        order_path = self.order_config['order_input_path'] / f"{order_id}_order.json"
        with open(order_path, 'w') as f:
            json.dump(asdict(order), f, indent=2)
        
        logger.info(f"Created order: {order_id}")
        
        return order


class RaspberryPiOptimizer:
    """Optimizations specific to Raspberry Pi 4B"""
    
    def __init__(self, config):
        self.config = config
        self.rpi_config = config.RPI_CONFIG
    
    def optimize_model(self, model_path: str, output_path: str = None) -> str:
        """
        Optimize model for Raspberry Pi inference
        
        Args:
            model_path: Path to trained model
            output_path: Path for optimized model
            
        Returns:
            Path to optimized model
        """
        if output_path is None:
            output_path = Path(model_path).parent / f"{Path(model_path).stem}_optimized.pt"
        
        logger.info("Optimizing model for Raspberry Pi 4B")
        
        # For YOLO models, export to ONNX for better CPU performance
        if self.rpi_config['use_onnx']:
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics not installed")
            
            model = YOLO(model_path)
            
            # Export to ONNX
            onnx_path = model.export(
                format='onnx',
                imgsz=self.config.DATASET_CONFIG['image_size'][0],
                simplify=True,
                dynamic=False
            )
            
            logger.info(f"Model exported to ONNX: {onnx_path}")
            return onnx_path
        
        # For TensorFlow Lite
        elif self.rpi_config['use_tflite']:
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics not installed")
            
            model = YOLO(model_path)
            
            # Export to TFLite
            tflite_path = model.export(
                format='tflite',
                imgsz=self.config.DATASET_CONFIG['image_size'][0],
                int8=self.rpi_config['quantization']
            )
            
            logger.info(f"Model exported to TFLite: {tflite_path}")
            return tflite_path
        
        else:
            logger.info("Using original PyTorch model")
            return model_path
    
    def set_cpu_threads(self, num_threads: int = None):
        """Set number of CPU threads for inference"""
        if num_threads is None:
            num_threads = self.rpi_config['thread_count']
        
        try:
            import torch
            torch.set_num_threads(num_threads)
            logger.info(f"Set CPU threads to {num_threads}")
        except Exception as e:
            logger.warning(f"Could not set CPU threads: {e}")
    
    def enable_memory_optimization(self):
        """Enable memory-efficient operations"""
        if self.rpi_config['memory_efficient']:
            try:
                import torch
                # Use smaller memory allocations
                torch.backends.cudnn.benchmark = False
                logger.info("Memory optimization enabled")
            except Exception as e:
                logger.warning(f"Could not enable memory optimization: {e}")
