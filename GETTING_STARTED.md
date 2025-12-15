# Getting Started with LEGO Assembly Error Detection System

## 🚀 Quick Start (5 Minutes)

This guide will help you set up and run your first assembly error detection in under 5 minutes.

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd lego_assembly_detection

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('Installation successful!')"
```

### Step 2: Prepare Your Dataset (1 minute)

Your dataset should have this structure:
```
your_data/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── labels/
    ├── image_001.txt  # Format: class_id x_center y_center width height
    ├── image_002.txt
    └── ...
```

Run preparation:
```bash
python main.py prepare \
    --images-path your_data/images \
    --labels-path your_data/labels \
    --output-path data/prepared_dataset \
    --kfold
```

### Step 3: Train Your First Model (Automated)

```bash
# This runs the complete pipeline: prepare → train → evaluate
python main.py pipeline \
    --images-path your_data/images \
    --labels-path your_data/labels
```

### Step 4: Test It! (1 minute)

```bash
# Process a single order
python main.py process \
    --model-path models/best_model.pt \
    --order-id TEST_001 \
    --model-type LEGO_MODEL \
    --photo-path path/to/test/image.jpg
```

✅ **Done!** You should see output like:
```
Result: RIGHT
Confidence: 0.92
Processing time: 0.234s
```

---

## 📖 Complete Workflow

### Workflow 1: Training from Rendered Images

```bash
# 1. Prepare dataset with K-fold
python main.py prepare \
    --images-path rendered_images/ \
    --labels-path rendered_labels/ \
    --kfold \
    --output-path data/prepared_dataset

# 2. Train with K-fold cross-validation
python main.py train \
    --dataset-path data/prepared_dataset \
    --use-kfold \
    --model-name lego_detector.pt

# 3. Evaluate model
python main.py evaluate \
    --model-path models/lego_detector.pt \
    --dataset-path data/prepared_dataset \
    --measure-inference
```

### Workflow 2: Fine-Tuning with Real Photos

```bash
# 1. Use your trained model from Workflow 1

# 2. Fine-tune with real photos
python main.py finetune \
    --model-path models/lego_detector.pt \
    --real-photos-path real_photos/ \
    --real-labels-path real_labels/

# 3. Evaluate fine-tuned model
python main.py evaluate \
    --model-path models/few_shot_finetuned.pt \
    --dataset-path data/prepared_dataset
```

### Workflow 3: Production Deployment

```bash
# 1. Optimize for Raspberry Pi
python main.py process \
    --model-path models/lego_detector.pt \
    --optimize-rpi \
    --order-id PROD_001 \
    --model-type LEGO_HOUSE \
    --photo-path captured_photo.jpg
```

---

## 🎯 Common Use Cases

### Use Case 1: Evaluate Multiple Models

```python
# compare_models.py
from main import *
import config

# Evaluate YOLOv8n vs YOLOv8s
args = type('Args', (), {
    'model_path': 'models/yolov8n_best.pt',
    'dataset_path': 'data/prepared_dataset',
    'alternative_models': ['yolov8s']
})()

logger = setup_logging(config.LOGS_DIR)
comparison = compare_models(args, logger)
```

### Use Case 2: Batch Process Orders

```python
# batch_process.py
from order_processor import OrderProcessor, OrderRequest
import config

processor = OrderProcessor(config)
processor.load_model('models/best_model.pt')

# Create batch
orders = []
for i in range(1, 101):
    order = OrderRequest(
        order_id=f"ORDER_{i:04d}",
        model_type="LEGO_HOUSE"
    )
    photo_path = f"photos/order_{i:04d}.jpg"
    orders.append((order, photo_path))

# Process batch
results = processor.batch_process_orders(orders)

# Print summary
correct = sum(1 for r in results if r.decision == "RIGHT")
print(f"Processed: {len(results)}, Correct: {correct}, Accuracy: {correct/len(results):.2%}")
```

### Use Case 3: Real-Time Monitoring

```python
# monitor.py
from order_processor import OrderProcessor
import config
import time
from pathlib import Path

processor = OrderProcessor(config)
processor.load_model('models/best_model.pt')

watch_dir = Path('incoming_photos')

print("Monitoring for new photos...")
while True:
    photos = list(watch_dir.glob('*.jpg'))
    
    for photo in photos:
        order_id = photo.stem
        order = OrderRequest(order_id=order_id, model_type="LEGO_MODEL")
        
        result = processor.process_order(order, str(photo))
        print(f"{order_id}: {result.decision} ({result.confidence:.2f})")
        
        # Move processed photo
        photo.rename(f"processed/{photo.name}")
    
    time.sleep(1)
```

---

## 🔧 Configuration Quick Reference

### Adjust Training Speed

```python
# In config.py
TRAINING_CONFIG = {
    'epochs': 50,        # Reduce for faster training
    'batch_size': 16,    # Increase if you have GPU
    'patience': 10,      # Reduce for faster early stopping
}
```

### Optimize for Accuracy

```python
TRAINING_CONFIG = {
    'epochs': 200,       # More epochs
    'patience': 30,      # More patience
}

DATASET_CONFIG = {
    'k_folds': 10,       # Use K-fold validation
}
```

### Optimize for Raspberry Pi Speed

```python
MODEL_CONFIG = {
    'variant': 'yolov8n',  # Use nano variant
}

DATASET_CONFIG = {
    'image_size': (320, 320),  # Smaller images
}

RPI_CONFIG = {
    'use_onnx': True,    # Enable ONNX export
    'thread_count': 2,    # Use 2 CPU threads
}
```

---

## 📊 Understanding Output

### Training Output
```
Epoch 1/100: 100%|████████| 50/50 [00:45<00:00,  1.11it/s]
train/box_loss: 0.234, train/cls_loss: 0.123
val/precision: 0.89, val/recall: 0.85, val/mAP50: 0.87
```

### Evaluation Output
```
EVALUATION METRICS SUMMARY
============================================================
Precision:     0.8945
Recall:        0.8623
F1-Score:      0.8781
mAP@0.5:       0.8756
mAP@0.5:0.95:  0.6234
------------------------------------------------------------
Inference Time: 234.5 ms
Total Time:     245.2 ms
============================================================
```

### Order Processing Output
```json
{
  "order_id": "ORDER_001",
  "decision": "WRONG",
  "confidence": 0.87,
  "detected_errors": ["assembly_error (conf: 0.87)"],
  "processing_time": 0.234,
  "timestamp": "2025-11-06T10:30:45.123456"
}
```

---

## 🐛 Troubleshooting

### Problem: "No module named 'ultralytics'"
**Solution:**
```bash
pip install ultralytics
```

### Problem: "CUDA out of memory"
**Solution:**
```python
# In config.py
TRAINING_CONFIG['batch_size'] = 4  # Reduce batch size
TRAINING_CONFIG['device'] = 'cpu'  # Use CPU instead
```

### Problem: "No valid image-label pairs found"
**Solution:**
- Check that image and label filenames match (except extension)
- Verify label format: `class_id x_center y_center width height`
- Ensure coordinates are normalized (0-1)

### Problem: Slow inference on Raspberry Pi
**Solution:**
```bash
# Use optimized model
python main.py process \
    --model-path models/yolov8n_best.pt \
    --optimize-rpi \
    ...
```

Or in config:
```python
RPI_CONFIG = {
    'use_onnx': True,
    'quantization': True,
}
DATASET_CONFIG['image_size'] = (320, 320)
```

---

## 📚 Next Steps

1. **Read the Full Documentation**
   - `README.md` - Complete system documentation
   - `DEPLOYMENT_GUIDE.md` - Raspberry Pi deployment
   - `PROJECT_SUMMARY.md` - Technical overview

2. **Explore Examples**
   - Run `python example_usage_complete.py` to see all features
   - Check individual example functions for specific use cases

3. **Customize Configuration**
   - Edit `config.py` to match your requirements
   - Adjust model variant, image size, training parameters

4. **Deploy to Production**
   - Follow `DEPLOYMENT_GUIDE.md` for Raspberry Pi setup
   - Set up monitoring and logging
   - Implement backup strategies

---

## 🎓 Learning Resources

### Understanding the System

1. **Data Flow:**
   ```
   Rendered Images → Dataset Preparation → Training → Evaluation
                                              ↓
   Real Photos → Few-Shot Fine-Tuning → Production Deployment
   ```

2. **Model Training Pipeline:**
   ```
   Raw Data → Preprocessing → Augmentation → Training → Validation → Testing
   ```

3. **Inference Pipeline:**
   ```
   Order Request → Load Image → Model Inference → Post-Processing → Decision
   ```

### Key Concepts

- **K-Fold Cross-Validation**: Splits data into K parts, trains K models, ensures robust evaluation
- **Few-Shot Learning**: Fine-tune with few real examples to adapt to real-world conditions
- **mAP (Mean Average Precision)**: Standard metric for object detection accuracy
- **IoU (Intersection over Union)**: Measures overlap between predicted and actual bounding boxes

---

## 💡 Tips for Best Results

1. **Data Quality**: Use high-quality images with good lighting and consistent camera angles
2. **Label Accuracy**: Ensure bounding boxes are tight and accurately placed
3. **Data Balance**: Have roughly equal numbers of correct and error examples
4. **Augmentation**: Enable augmentation for better generalization
5. **Validation**: Always use K-fold validation for reliable results
6. **Fine-Tuning**: Use few-shot fine-tuning when deploying to real-world scenarios
7. **Testing**: Test thoroughly on held-out test set before production deployment

---

## 📞 Getting Help

If you encounter issues:

1. **Check existing documentation** in README.md and DEPLOYMENT_GUIDE.md
2. **Review error messages** carefully - they often indicate the exact problem
3. **Check configuration** in config.py - many issues are configuration-related
4. **Try examples** in example_usage_complete.py to verify system is working
5. **Verify dataset** format and structure match requirements

---

## ✅ Checklist for Success

Before deploying to production, ensure:

- [ ] Dataset prepared with train/val/test splits
- [ ] K-fold cross-validation completed
- [ ] Model achieves >85% precision and recall
- [ ] Few-shot fine-tuning done with real photos
- [ ] Inference time meets requirements (<2s on target hardware)
- [ ] Raspberry Pi optimizations enabled
- [ ] Backup and monitoring systems in place
- [ ] Documentation reviewed and understood

---

**You're all set!** Start with the Quick Start section and work your way through the examples. Good luck with your LEGO assembly error detection system! 🎉
