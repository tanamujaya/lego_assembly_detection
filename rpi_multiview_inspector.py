#!/usr/bin/env python3
"""
LEGO Assembly Detection - Persistent Camera Version
Colors work correctly with this approach
"""

from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO
import time
from datetime import datetime
from pathlib import Path
import json

# Configuration
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = './models/best.pt'
OUTPUT_DIR = './inspections'

def run_inspection(model, camera, inspection_number, conf_threshold):
    """Run single inspection"""
    
    print("\n" + "="*60)
    print(f"INSPECTION #{inspection_number}")
    print(f"Confidence threshold: {conf_threshold}")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"model_{inspection_number:04d}"
    inspection_dir = Path(OUTPUT_DIR) / f"{model_id}_{timestamp}"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    
    angles = [0, 90, 180, 270]
    view_results = []
    
    for i, angle in enumerate(angles):
        print(f"\nView {i+1}/4: {angle} degrees")
        
        if i > 0:
            input(f"  Rotate model to {angle} degrees and press Enter...")
            time.sleep(2)
        else:
            input(f"  Place model at {angle} degrees and press Enter...")
        
        filename = inspection_dir / f"view_{angle}.jpg"
        print(f"  [Capturing...]")
        
        time.sleep(1)
        camera.capture_file(str(filename))
        print(f"  Saved: {filename.name}")
        
        print(f"  [Detecting...]")
        preds = model.predict(str(filename), conf=conf_threshold, verbose=False)
        boxes = preds[0].boxes
        
        if len(boxes) > 0:
            cls_id = int(boxes[0].cls[0])
            conf = float(boxes[0].conf[0])
            cls_name = 'correct' if cls_id == 0 else 'incorrect'
            status = "CORRECT" if cls_name == "correct" else "INCORRECT"
            print(f"  -> {angle} deg: {status} ({conf:.1%})")
            view_results.append({
                'angle': angle,
                'class_name': cls_name,
                'confidence': conf,
                'detected': True
            })
        else:
            print(f"  -> {angle} deg: No detection")
            view_results.append({
                'angle': angle,
                'class_name': 'correct',
                'confidence': 0.0,
                'detected': False
            })
            
# Decision
    print("\n" + "="*60)
    print("INSPECTION RESULTS")
    print("="*60)
    
    has_error = any(v['class_name'] == 'incorrect' and v['detected'] for v in view_results)
    
    if has_error:
        confidences = [v['confidence'] for v in view_results if v['detected']]
        avg_conf = max(confidences) if confidences else 0.0
        print(f"ASSEMBLY INCORRECT (confidence: {avg_conf:.1%})")
    else:
        confidences = [v['confidence'] for v in view_results if v['detected']]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        print(f"ASSEMBLY CORRECT (confidence: {avg_conf:.1%})")
    
    print(f"\nPer-View Results:")
    for v in view_results:
        status = "CORRECT" if v['class_name'] == 'correct' else "INCORRECT"
        conf_str = f"{v['confidence']:.1%}" if v['detected'] else "N/A"
        print(f"   {v['angle']:3d} deg: {status:9s} ({conf_str})")
    
    print(f"\nResults saved: {inspection_dir}")
    print("="*60)
    
    # Save
    results = {
        'model_id': model_id,
        'timestamp': datetime.now().isoformat(),
        'confidence_threshold': conf_threshold,
        'final_decision': 'incorrect' if has_error else 'correct',
        'view_results': view_results,
        'inspection_dir': str(inspection_dir)
    }
    
    with open(inspection_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def get_threshold_input(current_threshold):
    """Ask to change threshold"""
    print(f"\nCurrent threshold: {current_threshold}")
    response = input("Change threshold? (y/n or enter value): ").strip().lower()
    
    if response in ['n', 'no', '']:
        return current_threshold
    elif response in ['y', 'yes']:
        try:
            new = float(input("Enter new threshold (0.0-1.0): "))
            if 0.0 <= new <= 1.0:
                return new
        except:
            pass
    else:
        try:
            new = float(response)
            if 0.0 <= new <= 1.0:
                return new
        except:
            pass
    
    return current_threshold

# MAIN
print("="*60)
print("LEGO ASSEMBLY DETECTION SYSTEM")
print("="*60)
print("Initializing camera...")

camera = Picamera2()
config = camera.create_still_configuration(main={"size": (2464, 2464), "format": "RGB888"})
camera.configure(config)
camera.start()

print("Setting white balance to Fluorescent...")
camera.set_controls({
    "Brightness": 0.2,
    "Contrast": 1.2,
    "AwbMode": controls.AwbModeEnum.Fluorescent,
})

print("Waiting 10 seconds for stabilization...")
time.sleep(10)
print("Camera ready!")

model = YOLO(MODEL_PATH)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

inspection_counter = 1
current_threshold = CONFIDENCE_THRESHOLD

try:
    while True:
        results = run_inspection(model, camera, inspection_counter, current_threshold)
        
        print("\n" + "="*60)
        response = input("\nInspect another model? (y/n): ").strip().lower()
        
        if response in ['n', 'no', 'q', 'quit']:
            break
        elif response in ['y', 'yes', '']:
            inspection_counter += 1
            current_threshold = get_threshold_input(current_threshold)
        else:
            break

except KeyboardInterrupt:
    print("\n\nInterrupted")

finally:
    camera.stop()
    camera.close()
    print("\n" + "="*60)
    print(f"Total inspections: {inspection_counter - 1}")
    print("System shutdown complete")
    print("="*60)
