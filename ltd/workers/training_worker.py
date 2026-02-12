"""Worker thread for YOLO model training."""
import random
import shutil
from pathlib import Path

import yaml
from PySide6.QtCore import Signal

from ltd.workers.base_worker import BaseWorker


class TrainingWorker(BaseWorker):
    """Train a YOLO model on a dataset."""
    training_complete = Signal(str)  # path to trained model

    def __init__(self, dataset_path: str, model_path: str, model_name: str,
                 model_type: str, split_ratio: int, parent=None):
        super().__init__(parent)
        self.dataset_path = Path(dataset_path)
        self.model_path = model_path
        self.model_name = model_name
        self.model_type = model_type  # 'detect' or 'segment'
        self.split_ratio = split_ratio  # 50-95

    def do_work(self):
        import os
        import torch
        from ultralytics import YOLO

        self.status.emit('Preparing dataset...')

        # All paths must be absolute before we chdir
        models_dir = Path('./models/yolo').resolve()
        models_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.dataset_path.resolve()

        # Create temp training directory
        train_dir = self.dataset_path / 'train'
        val_dir = self.dataset_path / 'valid'

        # If train/valid don't exist, split the dataset
        if not train_dir.exists() or not val_dir.exists():
            self._split_dataset()

        # Count classes from label files
        classes = self._discover_classes()
        if not classes:
            self.error.emit('No classes found in dataset')
            return

        # Generate data.yaml if not present
        data_yaml_path = self.dataset_path / 'data.yaml'
        if not data_yaml_path.exists():
            data_config = {
                'path': str(self.dataset_path),
                'train': 'train/images',
                'val': 'valid/images',
                'nc': len(classes),
                'names': classes,
            }
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.status.emit(
            f'Training with {len(classes)} classes on {device}...')

        # Resolve model path: check models/yolo/ first for preset names
        model_file = Path(self.model_path)
        if not model_file.exists():
            local_path = models_dir / model_file.name
            if local_path.exists():
                model_file = local_path

        # chdir to models/yolo/ for the ENTIRE training process so any
        # model downloads (base model, pretrained backbone, etc.) land there
        # instead of polluting the app root.
        old_cwd = os.getcwd()
        os.chdir(str(models_dir))
        try:
            model = YOLO(str(model_file))

            task = self.model_type
            project_dir = str(self.dataset_path / 'runs' / task)

            results = model.train(
                data=str(data_yaml_path),
                epochs=500,
                patience=100,
                batch=-1,
                imgsz=640,
                device=device,
                cache='disk',
                deterministic=False,
                name=self.model_name,
                project=project_dir,
                exist_ok=True,
                verbose=True,
            )
        finally:
            os.chdir(old_cwd)

        # Copy best model to models/yolo
        best_model_path = (self.dataset_path / 'runs' / task /
                           self.model_name / 'weights' / 'best.pt')
        if best_model_path.exists():
            dest = models_dir / f'{self.model_name}.pt'
            shutil.copy2(best_model_path, dest)
            self.training_complete.emit(str(dest))
            self.status.emit(f'Training complete! Model saved to {dest}')
        else:
            self.status.emit('Training complete but no best.pt found')

    def _split_dataset(self):
        """Split flat dataset into train/valid structure."""
        images_dir = self.dataset_path / 'images'
        labels_dir = self.dataset_path / 'labels'

        # If there's no images subfolder, try the root
        if not images_dir.exists():
            images_dir = self.dataset_path
            labels_dir = self.dataset_path

        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])

        if not image_files:
            return

        random.shuffle(image_files)
        split_idx = int(len(image_files) * self.split_ratio / 100)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]

        for subset_name, subset_files in [('train', train_images),
                                           ('valid', val_images)]:
            img_dir = self.dataset_path / subset_name / 'images'
            lbl_dir = self.dataset_path / subset_name / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in subset_files:
                shutil.copy2(img_path, img_dir / img_path.name)
                label_file = labels_dir / f'{img_path.stem}.txt'
                if label_file.exists():
                    shutil.copy2(label_file, lbl_dir / label_file.name)

    def _discover_classes(self) -> list[str]:
        """Find class IDs in label files and generate class names."""
        class_ids = set()
        for labels_dir in [self.dataset_path / 'train' / 'labels',
                           self.dataset_path / 'valid' / 'labels',
                           self.dataset_path / 'labels',
                           self.dataset_path]:
            if not labels_dir.exists():
                continue
            for txt_file in labels_dir.glob('*.txt'):
                if txt_file.name == 'classes.txt':
                    # Read classes.txt if it exists
                    return [line.strip() for line in
                            txt_file.read_text().splitlines() if line.strip()]
                for line in txt_file.read_text().splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_ids.add(int(parts[0]))
                        except ValueError:
                            pass

        if class_ids:
            max_id = max(class_ids)
            return [f'class_{i}' for i in range(max_id + 1)]
        return []
