"""
Utility Script for Data Validation and System Testing
Helps validate dataset and test system functionality
"""

import sys
from pathlib import Path
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple

from config import Config


class DataValidator:
    """Validate dataset format and integrity"""
    
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.validation_report = {
            'images': {'total': 0, 'valid': 0, 'invalid': []},
            'labels': {'total': 0, 'valid': 0, 'invalid': []},
            'mismatches': [],
            'class_distribution': {},
            'image_sizes': [],
            'warnings': []
        }
    
    def validate_dataset(self, dataset_dir: Path) -> Dict:
        """
        Validate dataset structure and format
        
        Args:
            dataset_dir: Path to dataset directory
            
        Returns:
            Validation report dictionary
        """
        print(f"\n{'='*60}")
        print(f"Validating Dataset: {dataset_dir}")
        print(f"{'='*60}\n")
        
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        
        # Check directories exist
        if not images_dir.exists():
            print(f"❌ Error: Images directory not found: {images_dir}")
            return self.validation_report
        
        if not labels_dir.exists():
            print(f"❌ Error: Labels directory not found: {labels_dir}")
            return self.validation_report
        
        # Validate images
        print("Validating images...")
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        self.validation_report['images']['total'] = len(image_files)
        
        for img_path in image_files:
            if self._validate_image(img_path):
                self.validation_report['images']['valid'] += 1
            else:
                self.validation_report['images']['invalid'].append(str(img_path))
        
        # Validate labels
        print("Validating labels...")
        label_files = list(labels_dir.glob('*.txt'))
        self.validation_report['labels']['total'] = len(label_files)
        
        for label_path in label_files:
            if self._validate_label(label_path):
                self.validation_report['labels']['valid'] += 1
            else:
                self.validation_report['labels']['invalid'].append(str(label_path))
        
        # Check for mismatches
        print("Checking image-label pairs...")
        self._check_pairs(images_dir, labels_dir)
        
        # Calculate class distribution
        self._calculate_class_distribution(labels_dir)
        
        # Print report
        self._print_validation_report()
        
        return self.validation_report
    
    def _validate_image(self, img_path: Path) -> bool:
        """Validate single image file"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            self.validation_report['image_sizes'].append((w, h))
            
            # Check reasonable size
            if w < 100 or h < 100:
                self.validation_report['warnings'].append(
                    f"Small image size: {img_path.name} ({w}x{h})"
                )
            
            return True
        except Exception as e:
            self.validation_report['warnings'].append(
                f"Error loading image {img_path.name}: {str(e)}"
            )
            return False
    
    def _validate_label(self, label_path: Path) -> bool:
        """Validate single label file"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    self.validation_report['warnings'].append(
                        f"Invalid label format in {label_path.name}: {line}"
                    )
                    return False
                
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                
                # Validate ranges
                if class_id not in [0, 1]:
                    self.validation_report['warnings'].append(
                        f"Invalid class ID in {label_path.name}: {class_id}"
                    )
                    return False
                
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    self.validation_report['warnings'].append(
                        f"Invalid bbox coordinates in {label_path.name}: {x} {y} {w} {h}"
                    )
                    return False
            
            return True
        except Exception as e:
            self.validation_report['warnings'].append(
                f"Error reading label {label_path.name}: {str(e)}"
            )
            return False
    
    def _check_pairs(self, images_dir: Path, labels_dir: Path):
        """Check for matching image-label pairs"""
        image_stems = {p.stem for p in images_dir.glob('*') if p.suffix in ['.jpg', '.png']}
        label_stems = {p.stem for p in labels_dir.glob('*.txt')}
        
        # Images without labels
        images_without_labels = image_stems - label_stems
        if images_without_labels:
            self.validation_report['mismatches'].extend([
                f"Image without label: {stem}" for stem in images_without_labels
            ])
        
        # Labels without images
        labels_without_images = label_stems - image_stems
        if labels_without_images:
            self.validation_report['mismatches'].extend([
                f"Label without image: {stem}" for stem in labels_without_images
            ])
    
    def _calculate_class_distribution(self, labels_dir: Path):
        """Calculate distribution of classes in dataset"""
        class_counts = {0: 0, 1: 0}
        total_annotations = 0
        
        for label_path in labels_dir.glob('*.txt'):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_annotations += 1
            except:
                continue
        
        self.validation_report['class_distribution'] = {
            'correct': class_counts[0],
            'error': class_counts[1],
            'total_annotations': total_annotations
        }
    
    def _print_validation_report(self):
        """Print formatted validation report"""
        report = self.validation_report
        
        print(f"\n{'='*60}")
        print("VALIDATION REPORT")
        print(f"{'='*60}\n")
        
        # Images
        print("Images:")
        print(f"  Total: {report['images']['total']}")
        print(f"  Valid: {report['images']['valid']}")
        print(f"  Invalid: {len(report['images']['invalid'])}")
        
        # Labels
        print("\nLabels:")
        print(f"  Total: {report['labels']['total']}")
        print(f"  Valid: {report['labels']['valid']}")
        print(f"  Invalid: {len(report['labels']['invalid'])}")
        
        # Mismatches
        if report['mismatches']:
            print(f"\n⚠️  Mismatches found: {len(report['mismatches'])}")
            for mismatch in report['mismatches'][:5]:  # Show first 5
                print(f"  - {mismatch}")
            if len(report['mismatches']) > 5:
                print(f"  ... and {len(report['mismatches']) - 5} more")
        else:
            print("\n✓ All image-label pairs matched")
        
        # Class distribution
        print("\nClass Distribution:")
        dist = report['class_distribution']
        if dist:
            total = dist['total_annotations']
            print(f"  Total annotations: {total}")
            print(f"  Correct (class 0): {dist['correct']} ({dist['correct']/total*100:.1f}%)")
            print(f"  Error (class 1): {dist['error']} ({dist['error']/total*100:.1f}%)")
        
        # Image sizes
        if report['image_sizes']:
            sizes = np.array(report['image_sizes'])
            print(f"\nImage Size Statistics:")
            print(f"  Mean: {sizes.mean(axis=0).astype(int)}")
            print(f"  Min: {sizes.min(axis=0)}")
            print(f"  Max: {sizes.max(axis=0)}")
        
        # Warnings
        if report['warnings']:
            print(f"\n⚠️  Warnings: {len(report['warnings'])}")
            for warning in report['warnings'][:10]:  # Show first 10
                print(f"  - {warning}")
            if len(report['warnings']) > 10:
                print(f"  ... and {len(report['warnings']) - 10} more")
        
        # Summary
        print(f"\n{'='*60}")
        if (report['images']['valid'] == report['images']['total'] and 
            report['labels']['valid'] == report['labels']['total'] and 
            not report['mismatches'] and
            not report['warnings']):
            print("✅ Dataset validation PASSED")
        else:
            print("⚠️  Dataset validation completed with issues")
        print(f"{'='*60}\n")


class SystemTester:
    """Test system functionality"""
    
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
    
    def test_installation(self) -> bool:
        """Test if all dependencies are installed"""
        print(f"\n{'='*60}")
        print("Testing Installation")
        print(f"{'='*60}\n")
        
        required_packages = [
            ('ultralytics', 'YOLO'),
            ('torch', 'PyTorch'),
            ('cv2', 'OpenCV'),
            ('sklearn', 'scikit-learn'),
            ('matplotlib', 'Matplotlib'),
            ('seaborn', 'Seaborn'),
            ('yaml', 'PyYAML'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas')
        ]
        
        all_installed = True
        
        for package, name in required_packages:
            try:
                __import__(package)
                print(f"✓ {name} installed")
            except ImportError:
                print(f"❌ {name} NOT installed")
                all_installed = False
        
        print(f"\n{'='*60}")
        if all_installed:
            print("✅ All dependencies installed successfully")
        else:
            print("⚠️  Some dependencies missing. Run: pip install -r requirements.txt")
        print(f"{'='*60}\n")
        
        return all_installed
    
    def test_directories(self) -> bool:
        """Test if all required directories exist"""
        print(f"\n{'='*60}")
        print("Testing Directory Structure")
        print(f"{'='*60}\n")
        
        self.config.setup_directories()
        
        required_dirs = [
            self.config.DATA_DIR,
            self.config.MODELS_DIR,
            self.config.RESULTS_DIR,
            self.config.LOGS_DIR
        ]
        
        all_exist = True
        
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✓ {dir_path.name}/ exists")
            else:
                print(f"❌ {dir_path.name}/ does NOT exist")
                all_exist = False
        
        print(f"\n{'='*60}")
        if all_exist:
            print("✅ All directories exist")
        else:
            print("⚠️  Creating missing directories...")
            self.config.setup_directories()
        print(f"{'='*60}\n")
        
        return all_exist
    
    def test_model_loading(self, model_type: str = 'yolov8n') -> bool:
        """Test if model can be loaded"""
        print(f"\n{'='*60}")
        print(f"Testing Model Loading: {model_type}")
        print(f"{'='*60}\n")
        
        try:
            from model_manager import ModelManager
            
            manager = ModelManager(self.config)
            model = manager.get_model(model_type)
            model.load_model()
            
            print(f"✓ {model_type} loaded successfully")
            print(f"✓ Device: {model.device}")
            
            print(f"\n{'='*60}")
            print("✅ Model loading test PASSED")
            print(f"{'='*60}\n")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print(f"\n{'='*60}")
            print("⚠️  Model loading test FAILED")
            print(f"{'='*60}\n")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all system tests"""
        print("\n" + "="*60)
        print("RUNNING SYSTEM TESTS")
        print("="*60)
        
        tests = [
            ("Installation", self.test_installation),
            ("Directories", self.test_directories),
            ("Model Loading", lambda: self.test_model_loading())
        ]
        
        results = {}
        for test_name, test_func in tests:
            results[test_name] = test_func()
        
        print(f"\n{'='*60}")
        print("SYSTEM TEST SUMMARY")
        print(f"{'='*60}")
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        
        print(f"{'='*60}")
        if all_passed:
            print("✅ All system tests PASSED - System ready!")
        else:
            print("⚠️  Some tests FAILED - Please fix issues before proceeding")
        print(f"{'='*60}\n")
        
        return all_passed


def main():
    """Main utility function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Validation and System Testing Utility')
    parser.add_argument('command', choices=['validate', 'test', 'all'],
                       help='Command to execute')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset directory to validate')
    parser.add_argument('--save-report', action='store_true',
                       help='Save validation report to JSON')
    
    args = parser.parse_args()
    config = Config()
    
    if args.command == 'validate':
        if not args.dataset:
            print("Error: --dataset required for validation")
            return
        
        validator = DataValidator(config)
        report = validator.validate_dataset(Path(args.dataset))
        
        if args.save_report:
            report_path = config.RESULTS_DIR / 'validation_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Validation report saved to: {report_path}")
    
    elif args.command == 'test':
        tester = SystemTester(config)
        tester.run_all_tests()
    
    elif args.command == 'all':
        # Run system tests
        tester = SystemTester(config)
        system_ok = tester.run_all_tests()
        
        # Validate dataset if provided
        if args.dataset and system_ok:
            validator = DataValidator(config)
            validator.validate_dataset(Path(args.dataset))


if __name__ == '__main__':
    main()
