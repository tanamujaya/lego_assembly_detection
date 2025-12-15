# MULTI-VIEW LEGO DETECTION - QUICK REFERENCE

## 📋 Your Dataset Structure
- **Total images**: 6,400
- **Total models**: ~1,200
- **Views per model**: 4 (0°, 90°, 180°, 270°)
- **Naming**: `image_NNNN_AAA.png` (NNNN=sequential, AAA=angle)

## 🎯 Training (No Changes Needed!)

### Standard Training Command:
```bash
python main.py train \
  --mode kfold \
  --dataset /path/to/renamed_renders \
  --model yolov8n \
  --k 10 \
  --epochs 100
```

**Why no changes?** Each view is trained as an independent sample. Multi-view logic is applied during deployment only.

---

## 🚀 Deployment - Multi-View Inference

### 1. Single Model Inspection:

```python
from multiview_inference import MultiViewInspector

# Initialize
inspector = MultiViewInspector(
    model_path='./models/best.pt',
    decision_strategy='any_error',  # Recommended
    save_results=True
)

# Inspect model
image_paths = {
    0:   'path/to/image_0000_0.png',
    90:  'path/to/image_0000_90.png',
    180: 'path/to/image_0000_180.png',
    270: 'path/to/image_0000_270.png'
}

result = inspector.inspect_assembly_multiview(
    image_paths=image_paths,
    model_id='model_0000'
)

print(f"Decision: {result['final_decision_label']}")
```

### 2. Batch Evaluation:

```python
from multiview_inference import MultiViewBatchEvaluator

evaluator = MultiViewBatchEvaluator(
    model_path='./models/best.pt',
    decision_strategy='any_error'
)

results = evaluator.evaluate_test_set(
    images_dir='./data/test/images',
    labels_dir='./data/test/labels',
    output_path='./evaluation_results.json'
)

print(f"Model Accuracy: {results['model_accuracy']:.2%}")
```

---

## 🎛️ Decision Strategies

| Strategy | Logic | When to Use |
|----------|-------|-------------|
| **any_error** ⭐ | ANY view shows error → INCORRECT | High safety (recommended) |
| **majority_vote** | >50% views show error → INCORRECT | Balanced approach |
| **all_error** | ALL views show error → INCORRECT | Minimize false positives |

**Recommendation**: Use `any_error` for assembly verification

---

## 📊 Expected Performance

### Per-View Accuracy
- **Good**: > 85%
- **Excellent**: > 90%

### Model-Level Accuracy (with any_error)
- **Expected**: > 92%
- **Why higher?** Multiple views provide redundancy

---

## 🤖 Raspberry Pi Production System

### Hardware Setup:
- Raspberry Pi 4B (4GB+ RAM)
- Pi Camera Module
- Stepper motor + turntable
- GPIO connections

### Quick Deployment:
```python
from multiview_inference import MultiViewInspector

class ProductionSystem:
    def __init__(self):
        self.inspector = MultiViewInspector(
            model_path='./best.pt',
            decision_strategy='any_error'
        )
        # Add turntable controller
        # Add camera controller
    
    def inspect_model(self, model_id):
        # 1. Rotate to 0°, capture
        # 2. Rotate to 90°, capture
        # 3. Rotate to 180°, capture
        # 4. Rotate to 270°, capture
        # 5. Run multi-view inference
        # 6. Return decision
        pass
```

See `MULTIVIEW_GUIDE.py` for complete implementation.

---

## 📁 Files Provided

1. **multiview_config.py** - Configuration and helper functions
2. **multiview_inference.py** - Inference and evaluation classes
3. **MULTIVIEW_GUIDE.py** - Comprehensive guide with examples

---

## ⚡ Quick Tips

### Training:
- Use 320x320 resolution for speed (faster on RPi)
- YOLOv8n is recommended for Raspberry Pi
- 100-150 epochs is usually sufficient

### Deployment:
- Test on static images first before adding turntable
- Verify all 4 views before making decision
- Save results for later analysis

### Troubleshooting:
- **Low accuracy?** → Increase epochs, check labels
- **One view poor?** → Check lighting, camera position
- **Too many false positives?** → Try majority_vote
- **Slow on RPi?** → Use yolov8n, reduce resolution

---

## 🎓 Workflow Summary

```
Dataset (6400 images) 
    ↓
Train with existing system (treats each view independently)
    ↓
Model trained (best.pt)
    ↓
Deploy with multi-view logic (aggregate 4 views)
    ↓
Final decision: CORRECT or INCORRECT
```

---

## 📞 Need Help?

1. **Training issues**: Use your existing training system - it's already correct!
2. **Deployment questions**: See examples in `multiview_inference.py`
3. **Evaluation**: Use `MultiViewBatchEvaluator` class
4. **Hardware**: See production system example in `MULTIVIEW_GUIDE.py`

---

**Ready to start?** Your dataset is perfect, training needs no changes, just add the multi-view deployment scripts! 🚀
