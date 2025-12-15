"""
MULTI-VIEW TRAINING AND DEPLOYMENT GUIDE
For LEGO Assembly Error Detection System
"""

# ============================================================================
# OVERVIEW
# ============================================================================

"""
You have successfully renamed your dataset with multi-view structure:
- 6400 total images
- Pattern 1 (image_0000 to image_3199): 10 images per view
- Pattern 2 (image_3200 to image_6399): 2 images per view
- Each image labeled with view angle: _0, _90, _180, _270
- Total models: ~1200 (800 from pattern 1, 400 from pattern 2)

This guide explains how to:
1. Train your model on this multi-view dataset
2. Deploy for multi-view inference on Raspberry Pi
3. Evaluate multi-view system performance
"""

# ============================================================================
# STEP 1: TRAINING (No Changes Needed!)
# ============================================================================

"""
GOOD NEWS: Your existing training system already handles multi-view perfectly!

WHY? Because each view is trained as an independent sample. The model learns:
- "Is THIS specific view correct or incorrect?"
- View-specific error patterns
- Features visible from different angles

Your training treats:
  image_0000_0.png    → Sample 1 (class: correct/incorrect)
  image_0000_90.png   → Sample 2 (class: correct/incorrect)
  image_0000_180.png  → Sample 3 (class: correct/incorrect)
  image_0000_270.png  → Sample 4 (class: correct/incorrect)

This is exactly what we want!
"""

## TRAINING COMMAND (Same as before):

# Option 1: K-Fold Cross-Validation (Recommended for research)
"""
python main.py train \
  --mode kfold \
  --dataset /path/to/renamed_renders \
  --model yolov8n \
  --k 10 \
  --epochs 100 \
  --batch-size 16
"""

# Option 2: Standard Train/Val/Test Split
"""
python main.py train \
  --mode standard \
  --dataset /path/to/renamed_renders \
  --model yolov8n \
  --epochs 100 \
  --batch-size 16
"""

## TRAINING TIPS:

training_tips = {
    'image_size': '320x320 recommended (faster on RPi, still accurate)',
    'batch_size': '8-16 depending on your RAM',
    'epochs': '100-150 for good convergence',
    'model_choice': 'yolov8n for RPi deployment (fastest)',
    'augmentation': 'Keep enabled for better generalization',
}

## EXPECTED TRAINING TIME:

training_time_estimates = {
    'GPU (RTX 3080)': {
        'yolov8n': '~2-3 hours for 100 epochs',
        'yolov8s': '~4-5 hours for 100 epochs',
    },
    'Google Colab (Tesla T4)': {
        'yolov8n': '~4-6 hours for 100 epochs',
        'yolov8s': '~8-10 hours for 100 epochs',
    },
    'CPU': 'Not recommended (45+ hours)'
}

# ============================================================================
# STEP 2: DEPLOYMENT - MULTI-VIEW INFERENCE
# ============================================================================

"""
IMPORTANT: This is where multi-view logic comes in!

During deployment, you:
1. Capture/load 4 views of the same model
2. Run inference on each view independently
3. Aggregate predictions using a decision strategy
4. Make final decision: CORRECT or INCORRECT
"""

## DEPLOYMENT SETUP:

"""
1. Install the multi-view scripts on your deployment system:
   - multiview_config.py
   - multiview_inference.py

2. Place them in your lego_assembly_detection/ directory
"""

## SINGLE MODEL INSPECTION:

single_inspection_example = '''
from multiview_inference import MultiViewInspector

# Initialize inspector
inspector = MultiViewInspector(
    model_path='./models/yolov8n_train/weights/best.pt',
    confidence_threshold=0.5,
    decision_strategy='any_error',  # If ANY view shows error → INCORRECT
    save_results=True,
    output_dir='./inspection_results'
)

# Prepare image paths for 4 views
image_paths = {
    0:   '/path/to/test/images/image_0000_0.png',
    90:  '/path/to/test/images/image_0000_90.png',
    180: '/path/to/test/images/image_0000_180.png',
    270: '/path/to/test/images/image_0000_270.png'
}

# Run inspection
result = inspector.inspect_assembly_multiview(
    image_paths=image_paths,
    model_id='test_model_0000'
)

# Check result
print(f"Final Decision: {result['final_decision_label']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"View Breakdown:")
for view in result['view_results']:
    print(f"  {view['view_angle']}°: {view['class_name']} ({view['confidence']:.2f})")

# Result will be saved to: ./inspection_results/INS_test_model_0000/
# - result.json: Detailed JSON results
# - view_0.png, view_90.png, ...: Annotated individual views
# - summary_4view.png: 4-panel summary image
'''

## DECISION STRATEGIES:

decision_strategies = {
    'any_error': {
        'logic': 'If ANY view predicts error → Model = INCORRECT',
        'use_case': 'High safety requirement (recommended)',
        'example': '3 correct + 1 error → INCORRECT',
        'pros': ['Maximum error detection', 'Safest approach'],
        'cons': ['Higher false positive rate']
    },
    
    'majority_vote': {
        'logic': 'If >50% of views predict error → Model = INCORRECT',
        'use_case': 'Balanced accuracy and safety',
        'example': '2 correct + 2 error → INCORRECT',
        'pros': ['Balanced approach', 'Reduces false positives'],
        'cons': ['May miss some errors']
    },
    
    'all_error': {
        'logic': 'If ALL views predict error → Model = INCORRECT',
        'use_case': 'Minimize false positives',
        'example': '1 correct + 3 error → CORRECT',
        'pros': ['Lowest false positive rate'],
        'cons': ['May miss errors', 'Not recommended for safety-critical']
    }
}

## RECOMMENDATION:
recommended_strategy = 'any_error'  # Best for LEGO assembly verification

# ============================================================================
# STEP 3: BATCH EVALUATION ON TEST SET
# ============================================================================

"""
After training, evaluate your multi-view system on the test set to get:
- Model-level accuracy (entire 4-view models)
- Per-view accuracy (individual views)
- View performance comparison
"""

batch_evaluation_example = '''
from multiview_inference import MultiViewBatchEvaluator

# Initialize evaluator
evaluator = MultiViewBatchEvaluator(
    model_path='./models/yolov8n_train/weights/best.pt',
    confidence_threshold=0.5,
    decision_strategy='any_error'
)

# Evaluate on test set
results = evaluator.evaluate_test_set(
    images_dir='./data/prepared_dataset/test/images',
    labels_dir='./data/prepared_dataset/test/labels',
    output_path='./results/multiview_evaluation.json'
)

# Results include:
# - Model-level accuracy: % of complete models correctly classified
# - Per-view accuracy: Accuracy of each view angle (0°, 90°, 180°, 270°)
# - Detailed per-model results
'''

## EXPECTED METRICS:

expected_metrics = {
    'Per-View Accuracy': {
        'description': 'How often each individual view is correct',
        'good_threshold': '> 85%',
        'excellent_threshold': '> 90%'
    },
    'Model-Level Accuracy': {
        'description': 'How often the final multi-view decision is correct',
        'with_any_error_strategy': {
            'expected': '> 92% (higher than individual views)',
            'reason': 'Multiple views provide redundancy'
        }
    },
    'View Agreement': {
        'description': 'How often all 4 views agree',
        'interpretation': {
            'high_agreement': 'Model easy to classify',
            'low_agreement': 'Model difficult or ambiguous'
        }
    }
}

# ============================================================================
# STEP 4: RASPBERRY PI DEPLOYMENT WITH TURNTABLE
# ============================================================================

"""
For production deployment on Raspberry Pi with automated turntable:
"""

raspberry_pi_deployment = '''
# Hardware Setup:
# - Raspberry Pi 4B (4GB or 8GB)
# - Pi Camera Module or USB Camera
# - Stepper motor + driver for turntable
# - Power supply for motor

# Software Setup:
import RPi.GPIO as GPIO
import time
import cv2
from multiview_inference import MultiViewInspector

class TurntableController:
    """Controls stepper motor for rotating LEGO model"""
    
    def __init__(self, step_pin=17, dir_pin=27, enable_pin=22):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(step_pin, GPIO.OUT)
        GPIO.setup(dir_pin, GPIO.OUT)
        GPIO.setup(enable_pin, GPIO.OUT)
        
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.enable_pin = enable_pin
        
        GPIO.output(enable_pin, GPIO.LOW)  # Enable motor
    
    def rotate_to_angle(self, angle):
        """Rotate turntable to specific angle (0, 90, 180, 270)"""
        # Calculate steps (adjust for your motor/gearing)
        steps_per_degree = (200 * 16) / 360  # 200 steps/rev, 16x microstepping
        steps = int(angle * steps_per_degree)
        
        # Set direction
        GPIO.output(self.dir_pin, GPIO.HIGH)
        
        # Rotate
        for _ in range(abs(steps)):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(0.001)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(0.001)
        
        time.sleep(0.5)  # Wait for vibrations to settle
    
    def cleanup(self):
        GPIO.cleanup()


class ProductionInspectionSystem:
    """Complete inspection system with camera and turntable"""
    
    def __init__(self, model_path, output_dir='./production_results'):
        self.inspector = MultiViewInspector(
            model_path=model_path,
            confidence_threshold=0.5,
            decision_strategy='any_error',
            save_results=True,
            output_dir=output_dir
        )
        self.turntable = TurntableController()
        self.camera = cv2.VideoCapture(0)  # Pi Camera
        
        # Warm up camera
        for _ in range(10):
            self.camera.read()
    
    def capture_view(self, angle):
        """Rotate turntable and capture image at specified angle"""
        print(f"Rotating to {angle}°...")
        self.turntable.rotate_to_angle(angle)
        
        print(f"Capturing image at {angle}°...")
        time.sleep(0.3)  # Camera stabilization
        
        ret, frame = self.camera.read()
        if not ret:
            raise Exception(f"Failed to capture image at {angle}°")
        
        # Resize to match training resolution
        frame_resized = cv2.resize(frame, (320, 320))
        
        return frame_resized
    
    def inspect_model(self, model_id):
        """Perform complete 4-view inspection"""
        print(f"\\nStarting inspection of model {model_id}")
        
        # Capture all 4 views
        captured_images = {}
        temp_dir = Path(f'./temp_captures/{model_id}')
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for angle in [0, 90, 180, 270]:
            image = self.capture_view(angle)
            
            # Save temporarily
            image_path = temp_dir / f'view_{angle}.png'
            cv2.imwrite(str(image_path), image)
            captured_images[angle] = str(image_path)
        
        # Run multi-view inspection
        result = self.inspector.inspect_assembly_multiview(
            image_paths=captured_images,
            model_id=model_id
        )
        
        # Return to home position
        self.turntable.rotate_to_angle(0)
        
        return result
    
    def cleanup(self):
        self.camera.release()
        self.turntable.cleanup()


# Usage:
if __name__ == "__main__":
    system = ProductionInspectionSystem(
        model_path='./models/yolov8n_train/weights/best.pt'
    )
    
    try:
        # Inspect model
        result = system.inspect_model(model_id='production_001')
        
        # Take action based on result
        if result['final_decision_label'] == 'CORRECT':
            print("✅ PASS: Model approved for shipping")
            # TODO: Signal green light, move to next station
        else:
            print("❌ FAIL: Assembly error detected")
            # TODO: Signal red light, alert operator
            print(f"Error detected in views: {[v['view_angle'] for v in result['view_results'] if v['predicted_class'] == 1]}")
    
    finally:
        system.cleanup()
'''

# ============================================================================
# STEP 5: PERFORMANCE OPTIMIZATION
# ============================================================================

performance_tips = {
    'Training': {
        'use_smaller_images': 'Train with 320x320 or 224x224 (2-4x faster)',
        'use_gpu': 'Google Colab with T4 GPU is free',
        'reduce_batch_size': 'If running out of memory',
        'early_stopping': 'Enable patience=20 to stop if not improving'
    },
    
    'Inference_on_RPi': {
        'use_yolov8n': 'Smallest/fastest model (~0.5-1s per image)',
        'enable_fp16': 'Half precision for faster inference',
        'optimize_resolution': '320x320 or 224x224 for real-time',
        'use_threading': 'Process views in parallel if possible',
        'cache_model': 'Load model once at startup'
    },
    
    'Multi-View_Specific': {
        'capture_strategy': 'Continuous rotation vs stop-capture-rotate',
        'parallel_processing': 'Process all views together if memory allows',
        'early_termination': 'If 2+ views show error, can stop early with any_error strategy'
    }
}

# ============================================================================
# STEP 6: TROUBLESHOOTING
# ============================================================================

troubleshooting = {
    'Low_Accuracy': {
        'symptoms': 'Model accuracy < 80%',
        'solutions': [
            'Increase training epochs (150-200)',
            'Check data quality and labeling',
            'Ensure all 4 views are properly labeled',
            'Try different decision strategy',
            'Increase confidence threshold to reduce false positives'
        ]
    },
    
    'One_View_Performs_Poorly': {
        'symptoms': 'One angle consistently less accurate',
        'solutions': [
            'Check lighting for that view angle',
            'Ensure camera position is consistent',
            'May need more training data for that angle',
            'Could indicate model-specific issue visible from that angle'
        ]
    },
    
    'High_False_Positive_Rate': {
        'symptoms': 'Too many correct models flagged as errors',
        'solutions': [
            'Try majority_vote instead of any_error',
            'Increase confidence threshold',
            'Check if training data is imbalanced',
            'Add more correct model examples to training'
        ]
    },
    
    'Slow_Inference_on_RPi': {
        'symptoms': '> 2 seconds per model (4 views)',
        'solutions': [
            'Use yolov8n instead of yolov8s',
            'Reduce image resolution to 224x224',
            'Enable FP16 half precision',
            'Check CPU temperature (thermal throttling)',
            'Close other applications'
        ]
    }
}

# ============================================================================
# QUICK START CHECKLIST
# ============================================================================

quick_start_checklist = """
□ Dataset renamed with multi-view structure (DONE! ✓)
□ Dataset organized in YOLO format (images/ and labels/ folders)
□ Training system configured (use existing system)
□ Train model using your existing pipeline
□ Evaluate single-view performance on test set
□ Install multiview_config.py and multiview_inference.py
□ Test multi-view inference on few examples
□ Run batch evaluation on test set
□ Compare multi-view vs single-view accuracy
□ Deploy to Raspberry Pi
□ Set up turntable hardware (if automated)
□ Test production inspection system
□ Monitor and iterate on performance

ESTIMATED TIME TO DEPLOYMENT:
- Training: 2-6 hours (depending on GPU)
- Testing multi-view system: 1-2 hours
- Raspberry Pi setup: 2-4 hours
- Total: 1-2 days from training to production
"""

# ============================================================================
# SUMMARY
# ============================================================================

summary = """
CONGRATULATIONS! Your dataset is ready for multi-view training.

KEY POINTS:
1. ✅ Dataset is properly formatted with multi-view structure
2. ✅ NO changes needed to training pipeline
3. ✅ Multi-view logic applied during DEPLOYMENT only
4. ✅ Decision strategy: 'any_error' recommended for safety
5. ✅ Expected improvement: 5-10% better accuracy than single view

NEXT STEPS:
1. Train your model (use existing training pipeline)
2. Test multi-view inference with the provided scripts
3. Evaluate on test set to compare performance
4. Deploy to Raspberry Pi with turntable

QUESTIONS?
- Training: Use your existing system, it's already correct!
- Deployment: Use multiview_inference.py for 4-view logic
- Evaluation: Use MultiViewBatchEvaluator for testing
- Hardware: Setup turntable controller for automated rotation

YOU'RE READY TO GO! 🚀
"""

print(summary)
