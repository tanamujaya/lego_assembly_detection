# Getting Started Checklist

Use this checklist to set up and deploy your LEGO Assembly Error Detection System.

## Phase 1: Installation & Setup (15 minutes)

### Prerequisites
- [ ] Python 3.8+ installed
- [ ] pip package manager available
- [ ] Virtual environment tool (optional but recommended)
- [ ] 8GB+ RAM for training (4GB for inference only)
- [ ] GPU optional for faster training

### Installation Steps
- [ ] Navigate to project directory: `cd lego_assembly_detection`
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate virtual environment: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify installation: `python utils.py test`

**✓ Checkpoint**: All tests should pass. If not, check error messages and install missing packages.

## Phase 2: Data Preparation (30 minutes)

### Dataset Organization
- [ ] Create directory structure:
  ```
  data/
  ├── renders/
  │   ├── images/
  │   └── labels/
  └── real_photos/  (for later fine-tuning)
      ├── images/
      └── labels/
  ```

### Prepare Your Render Dataset
- [ ] Place rendered images in `data/renders/images/`
- [ ] Create YOLO format labels in `data/renders/labels/`
- [ ] Ensure each image has a corresponding .txt label file
- [ ] Label format: `class_id x_center y_center width height` (normalized 0-1)
- [ ] Classes: 0 = correct, 1 = error

### Validate Your Data
- [ ] Run validation: `python utils.py validate --dataset ./data/renders`
- [ ] Review validation report
- [ ] Fix any issues identified (missing labels, format errors, etc.)
- [ ] Re-run validation until all checks pass

**✓ Checkpoint**: Validation should show no errors. Note the class distribution.

## Phase 3: First Training Run (30-60 minutes)

### Quick Training (Standard Split)
- [ ] Run quick training: `python main.py train --mode standard --dataset ./data/renders --model yolov8n --epochs 30`
- [ ] Monitor training progress (watch for decreasing loss)
- [ ] Wait for training to complete
- [ ] Note the location of best model: `./models/yolov8n_train/weights/best.pt`

### Review Results
- [ ] Check training results in `./results/yolov8n_standard_results.json`
- [ ] Note test metrics: Precision, Recall, mAP@0.5
- [ ] Generate evaluation report: `python main.py evaluate --results ./results/yolov8n_standard_results.json`

**✓ Checkpoint**: Training should complete without errors. Test mAP@0.5 should be >0.70 for a good model.

## Phase 4: Test Inference (10 minutes)

### Single Image Test
- [ ] Prepare a test image (either from test set or new image)
- [ ] Run inference: `python main.py infer --model ./models/yolov8n_train/weights/best.pt --image test_image.jpg --visualize`
- [ ] Review results:
  - [ ] Result: Right or Wrong
  - [ ] Assessment time (should be <5s on CPU, <1s on GPU)
  - [ ] Confidence scores
- [ ] Check visualization: `./results/visualizations/`

### Batch Test (if you have multiple images)
- [ ] Prepare test images directory
- [ ] Run batch inference: `python main.py infer --model ./models/best.pt --batch-dir ./test_images`
- [ ] Review batch results and statistics

**✓ Checkpoint**: Inference should work correctly with reasonable assessment times.

## Phase 5: K-Fold Training (Optional but Recommended) (1-2 hours)

### Full K-Fold Cross-Validation
- [ ] Run K-fold training: `python main.py train --mode kfold --dataset ./data/renders --model yolov8n --k 10`
- [ ] This will take longer (10x the standard training time)
- [ ] Review aggregate metrics after completion
- [ ] Note best fold and model path

### Evaluate K-Fold Results
- [ ] Generate K-fold plots: `python main.py evaluate --results ./results/yolov8n_kfold_results.json --plot-kfold`
- [ ] Review metrics plot: Mean ± Std for each metric
- [ ] Check test set performance
- [ ] Compare with standard training results

**✓ Checkpoint**: K-fold should show consistent performance across folds (low std).

## Phase 6: Few-Shot Fine-Tuning (When Real Photos Available)

### Prepare Real Photos
- [ ] Collect 10-20 real photos of LEGO assemblies
- [ ] Label them in YOLO format
- [ ] Place in `data/real_photos/images/` and `data/real_photos/labels/`
- [ ] Validate: `python utils.py validate --dataset ./data/real_photos`

### Fine-Tune Model
- [ ] Run fine-tuning: `python main.py finetune --base-model ./models/best.pt --fewshot-data ./data/real_photos --epochs 50`
- [ ] Wait for fine-tuning to complete
- [ ] Note fine-tuned model path
- [ ] Test on real photos to verify improvement

**✓ Checkpoint**: Fine-tuned model should show improved performance on real photos.

## Phase 7: Model Comparison (Optional)

### Train Alternative Models
- [ ] Train YOLOv5n: `python main.py train --mode kfold --dataset ./data/renders --model yolov5n`
- [ ] Train YOLOv8s: `python main.py train --mode kfold --dataset ./data/renders --model yolov8s`

### Compare Models
- [ ] Run comparison: `python main.py compare --results ./results/*.json --dashboard`
- [ ] Review comparison charts
- [ ] Review performance dashboard
- [ ] Select best model for your use case

**✓ Checkpoint**: You should have clear understanding of which model performs best.

## Phase 8: Raspberry Pi Deployment

### Prepare Raspberry Pi
- [ ] Ensure RPi 4B with 4GB+ RAM
- [ ] Install Raspberry Pi OS (64-bit recommended)
- [ ] Install Python 3.8+: `sudo apt-get install python3-pip`
- [ ] Install OpenCV: `sudo apt-get install python3-opencv`

### Transfer Model
- [ ] Copy project to RPi: `scp -r lego_assembly_detection pi@raspberrypi:~`
- [ ] Or copy just the model: `scp ./models/best.pt pi@raspberrypi:~`

### Install Dependencies on RPi
- [ ] SSH into RPi: `ssh pi@raspberrypi`
- [ ] Navigate to project: `cd lego_assembly_detection`
- [ ] Install dependencies: `pip3 install -r requirements.txt`
- [ ] Verify: `python3 utils.py test`

### Test Inference on RPi
- [ ] Run test inference: `python3 main.py infer --model ~/best.pt --image test.jpg`
- [ ] Check assessment time (should be <1s for YOLOv8n)
- [ ] Verify results are correct

**✓ Checkpoint**: Inference should work on RPi with acceptable speed (<2s per image).

## Phase 9: Production Setup

### Set Up Production Pipeline
- [ ] Create production script or integrate with existing system
- [ ] Set up image capture (camera module or USB camera)
- [ ] Configure order request system (JSON file, database, API)
- [ ] Test end-to-end workflow

### Performance Optimization (if needed)
- [ ] Export to ONNX: `model.export_model(format='onnx')`
- [ ] Or TFLite: `model.export_model(format='tflite')`
- [ ] Test exported model
- [ ] Measure performance improvement

### Monitoring and Logging
- [ ] Set up result logging
- [ ] Configure error alerts
- [ ] Monitor assessment times
- [ ] Track accuracy metrics

**✓ Checkpoint**: Production system should be running smoothly.

## Phase 10: Continuous Improvement

### Data Collection
- [ ] Collect real production images
- [ ] Label any errors found
- [ ] Build dataset of real production cases

### Model Refinement
- [ ] Fine-tune with new real data
- [ ] Re-evaluate performance
- [ ] Update production model if improved

### System Maintenance
- [ ] Monitor performance metrics
- [ ] Review false positives/negatives
- [ ] Update training data as needed
- [ ] Retrain periodically with new data

**✓ Checkpoint**: System should be continuously improving with real-world data.

---

## Quick Reference Commands

### Training
```bash
# Standard training
python main.py train --mode standard --dataset ./data/renders --model yolov8n

# K-fold training
python main.py train --mode kfold --dataset ./data/renders --model yolov8n --k 10

# Few-shot fine-tuning
python main.py finetune --base-model ./models/best.pt --fewshot-data ./data/real_photos
```

### Inference
```bash
# Single image
python main.py infer --model ./models/best.pt --image test.jpg --visualize

# Batch processing
python main.py infer --model ./models/best.pt --batch-dir ./test_images
```

### Evaluation
```bash
# Evaluate results
python main.py evaluate --results ./results/results.json --plot-kfold

# Compare models
python main.py compare --results ./results/model1.json ./results/model2.json --dashboard
```

### Utilities
```bash
# Validate dataset
python utils.py validate --dataset ./data/renders

# Test system
python utils.py test
```

---

## Troubleshooting Checklist

### Installation Issues
- [ ] Check Python version: `python --version` (should be 3.8+)
- [ ] Update pip: `pip install --upgrade pip`
- [ ] Try installing packages one by one from requirements.txt
- [ ] Check for system-specific dependencies (CUDA for GPU, etc.)

### Training Issues
- [ ] Verify dataset paths are correct
- [ ] Check that labels are in correct format
- [ ] Reduce batch size if out of memory
- [ ] Reduce epochs for testing
- [ ] Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Inference Issues
- [ ] Verify model path exists
- [ ] Check image path is correct
- [ ] Verify image format (JPG or PNG)
- [ ] Try with a different image
- [ ] Check model was trained successfully

### Raspberry Pi Issues
- [ ] Ensure sufficient RAM (4GB minimum)
- [ ] Use smaller model (YOLOv8n or YOLOv5n)
- [ ] Close other applications
- [ ] Check CPU temperature
- [ ] Reduce image resolution if needed

---

## Success Criteria

### Minimum Viable System
- ✓ Training completes successfully
- ✓ Test mAP@0.5 > 0.70
- ✓ Inference works correctly
- ✓ Assessment time < 2s on target hardware
- ✓ System deployed and running

### Production Ready
- ✓ K-fold CV shows consistent performance
- ✓ Test mAP@0.5 > 0.85
- ✓ Fine-tuned on real photos
- ✓ Assessment time < 1s on Raspberry Pi
- ✓ Production pipeline integrated
- ✓ Monitoring and logging in place

### Optimized System
- ✓ Multiple models compared
- ✓ Best model selected and optimized
- ✓ Test mAP@0.5 > 0.90
- ✓ Assessment time < 0.5s on Raspberry Pi
- ✓ Continuous improvement pipeline established
- ✓ Real-world accuracy validated

---

## Next Steps After Completion

1. **Documentation**: Document your specific setup and any customizations
2. **Backup**: Back up trained models and training data
3. **Monitoring**: Set up performance monitoring dashboard
4. **Training**: Train team members on system operation
5. **Maintenance**: Schedule periodic model retraining
6. **Improvement**: Establish data collection and improvement cycle

---

**Remember**: Start simple, validate each step, and iterate. The system is designed to be incrementally deployable - you don't need to complete everything at once!

**Need Help?** Check INDEX.md for navigation or README.md for detailed documentation.

🚀 Good luck with your LEGO assembly detection system!
