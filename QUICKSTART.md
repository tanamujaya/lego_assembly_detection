# Quick Start Guide

Get up and running with the LEGO Assembly Error Detection System in minutes!

## Prerequisites

- Python 3.8 or higher
- Your dataset of LEGO images with labels

## Step-by-Step Setup

### 1. Install Dependencies (5 minutes)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install ultralytics torch torchvision opencv-python scikit-learn matplotlib seaborn pyyaml

# Verify installation
python -c "from ultralytics import YOLO; print('Installation successful!')"
```

### 2. Prepare Your Data (10 minutes)

Organize your dataset in this structure:

```
data/
└── renders/
    ├── images/
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    └── labels/
        ├── img_001.txt
        ├── img_002.txt
        └── ...
```

**Label format** (YOLO format):
```
# img_001.txt
1 0.5 0.3 0.2 0.15  # error at x=0.5, y=0.3, width=0.2, height=0.15
```

Classes:
- 0 = correct (no errors)
- 1 = error (assembly mistake)

### 3. Train Your First Model (30-60 minutes)

**Option A: Quick Training (Standard Split)**

```bash
python main.py train \
  --mode standard \
  --dataset ./data/renders \
  --model yolov8n \
  --epochs 50 \
  --batch-size 16
```

**Option B: Robust Training (K-Fold Cross-Validation)**

```bash
python main.py train \
  --mode kfold \
  --dataset ./data/renders \
  --model yolov8n \
  --k 10 \
  --epochs 100
```

Training will automatically:
- Split your data (70% train, 15% val, 15% test)
- Train the model
- Evaluate on test set
- Save results to `./results/`
- Save best model to `./models/`

### 4. Test Your Model (1 minute)

```bash
python main.py infer \
  --model ./models/yolov8n_train/weights/best.pt \
  --image ./test_image.jpg \
  --order '{"model_name": "Test Assembly"}' \
  --visualize
```

Output:
```
==================================================
Inspection ID: INS_20250106_143022
Order: Test Assembly
==================================================

Result: Wrong
Errors detected: 2
Assessment time: 0.687 seconds
==================================================
```

### 5. Deploy to Raspberry Pi (Optional)

**On your computer:**
```bash
# Copy model to Raspberry Pi
scp ./models/yolov8n_train/weights/best.pt pi@raspberrypi.local:~/model.pt
```

**On Raspberry Pi:**
```bash
# Install dependencies
pip3 install ultralytics opencv-python

# Run inference
python main.py infer \
  --model ~/model.pt \
  --image ./captured_image.jpg \
  --order '{"model_name": "Castle"}'
```

## Common Workflows

### Workflow 1: Train → Evaluate → Deploy

```bash
# 1. Train
python main.py train --mode standard --dataset ./data/renders --model yolov8n

# 2. Evaluate
python main.py evaluate --results ./results/yolov8n_standard_results.json --plot-kfold

# 3. Test batch inference
python main.py infer --model ./models/yolov8n_train/weights/best.pt --batch-dir ./test_images
```

### Workflow 2: Train on Renders → Fine-tune on Real Photos

```bash
# 1. Train on synthetic renders
python main.py train --mode kfold --dataset ./data/renders --model yolov8n

# 2. Fine-tune on real photos
python main.py finetune \
  --base-model ./models/yolov8n_fold1/weights/best.pt \
  --fewshot-data ./data/real_photos \
  --epochs 50

# 3. Deploy fine-tuned model
python main.py infer \
  --model ./models/yolov8n_finetuned/weights/best.pt \
  --image ./real_test_image.jpg
```

### Workflow 3: Compare Multiple Models

```bash
# Train multiple models
python main.py train --mode kfold --dataset ./data/renders --model yolov8n
python main.py train --mode kfold --dataset ./data/renders --model yolov5n

# Compare results
python main.py compare \
  --results ./results/yolov8n_kfold_results.json \
            ./results/yolov5n_kfold_results.json \
  --dashboard
```

## Python API Quick Examples

### Example 1: Simple Training

```python
from pathlib import Path
from config import Config
from training_pipeline import TrainingPipeline

# Setup
config = Config()
config.EPOCHS = 50
pipeline = TrainingPipeline(config)

# Train
results = pipeline.train_standard_split(
    model_type='yolov8n',
    dataset_dir=Path('./data/renders')
)

print(f"Test mAP50: {results['test_metrics']['mAP50']:.4f}")
```

### Example 2: Production Inference

```python
from config import Config
from inference import LEGOAssemblyInspector

# Initialize
inspector = LEGOAssemblyInspector(
    model_path='./models/yolov8n_train/weights/best.pt'
)

# Inspect
order = {'model_name': 'Castle', 'model_id': '12345'}
result = inspector.inspect_assembly(order, './test_image.jpg')

if result['result'] == 'Wrong':
    print(f"⚠️ Assembly error detected! Found {result['error_count']} errors")
else:
    print("✓ Assembly correct!")
```

### Example 3: Batch Processing

```python
from pathlib import Path
from inference import LEGOAssemblyInspector

inspector = LEGOAssemblyInspector(
    model_path='./models/yolov8n_train/weights/best.pt'
)

# Prepare batch
test_dir = Path('./test_images')
inspections = []
for img_path in test_dir.glob('*.jpg'):
    order = {'model_name': img_path.stem}
    inspections.append((order, str(img_path)))

# Process
results = inspector.batch_inspect(inspections)

# Statistics
stats = inspector.get_statistics()
print(f"Processed: {stats['total_inspections']}")
print(f"Average time: {stats['average_inference_time']:.3f}s")
```

## Expected Results

After training on a typical dataset (1000+ images):

| Metric | Expected Range | Good Performance |
|--------|----------------|------------------|
| Precision | 0.85 - 0.95 | > 0.90 |
| Recall | 0.80 - 0.92 | > 0.85 |
| mAP@0.5 | 0.85 - 0.95 | > 0.90 |
| Inference Time (RPi) | 0.5 - 1.5s | < 1.0s |

## Troubleshooting Quick Fixes

**Problem**: Out of memory during training
```bash
# Solution: Reduce batch size
python main.py train --dataset ./data/renders --model yolov8n --batch-size 8
```

**Problem**: Training too slow
```bash
# Solution: Reduce epochs or use smaller model
python main.py train --dataset ./data/renders --model yolov8n --epochs 30
```

**Problem**: Poor accuracy
```bash
# Solution 1: Train longer
python main.py train --dataset ./data/renders --model yolov8n --epochs 150

# Solution 2: Use larger model (if not deploying to RPi)
python main.py train --dataset ./data/renders --model yolov8s

# Solution 3: Fine-tune with real photos
python main.py finetune --base-model ./models/best.pt --fewshot-data ./data/real_photos
```

## Next Steps

1. **Optimize for Production**: Export to ONNX or TFLite for faster inference
2. **Data Augmentation**: Add more training data with variations
3. **Hyperparameter Tuning**: Experiment with learning rates and architectures
4. **Real-time Processing**: Set up camera pipeline for continuous inspection

## Need Help?

- Check the full README.md for detailed documentation
- Review example outputs in `./results/`
- Check training logs in `./models/your_model/`

## Minimal Working Example

The absolute minimum to get started:

```bash
# Install
pip install ultralytics opencv-python scikit-learn

# Train (assuming data is ready)
python main.py train --dataset ./data/renders --model yolov8n --epochs 30

# Test
python main.py infer \
  --model ./models/yolov8n_train/weights/best.pt \
  --image ./test.jpg
```

That's it! You're ready to detect LEGO assembly errors! 🎉
