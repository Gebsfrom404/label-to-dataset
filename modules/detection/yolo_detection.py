"""Built-in YOLO detection module."""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

from modules.base import BaseDetectionModule


class YoloDetectionModule(BaseDetectionModule):

    def __init__(self):
        self._model = None
        self._model_path = ''
        self._confidence = 0.5
        self._settings_widget = None
        self._model_path_label = None
        self._conf_label = None
        self._restore_settings()

    @property
    def name(self) -> str:
        return 'YOLO Model'

    def get_settings_widget(self) -> QWidget:
        if self._settings_widget is not None:
            return self._settings_widget

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Model path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel('Model:'))
        label_text = (Path(self._model_path).name if self._model_path
                      else 'No model selected')
        self._model_path_label = QLabel(label_text)
        self._model_path_label.setWordWrap(True)
        path_layout.addWidget(self._model_path_label, stretch=1)
        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Confidence slider with value label
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel('Confidence:'))
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(1, 99)
        self._conf_slider.setValue(int(self._confidence * 100))
        self._conf_slider.setTickInterval(10)
        self._conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._conf_slider.valueChanged.connect(self._on_conf_slider_changed)
        conf_layout.addWidget(self._conf_slider, stretch=1)
        self._conf_label = QLabel(f'{self._confidence:.2f}')
        self._conf_label.setMinimumWidth(35)
        conf_layout.addWidget(self._conf_label)
        layout.addLayout(conf_layout)

        layout.addStretch()
        self._settings_widget = widget
        return widget

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            None, 'Select YOLO Model', './models/yolo',
            'YOLO Models (*.pt *.onnx);;All Files (*)')
        if path:
            self._model_path = path
            self._model_path_label.setText(Path(path).name)
            self._model = None  # Reset model
            self._save_settings()

    def _on_conf_slider_changed(self, value):
        self._confidence = value / 100.0
        if self._conf_label:
            self._conf_label.setText(f'{self._confidence:.2f}')
        self._save_settings()

    def _save_settings(self):
        from ltd.settings import get_settings
        settings = get_settings()
        settings.setValue('yolo_detection/model_path', self._model_path)
        settings.setValue('yolo_detection/confidence', self._confidence)

    def _restore_settings(self):
        from ltd.settings import get_settings
        settings = get_settings()
        path = settings.value('yolo_detection/model_path', '', type=str)
        if path and Path(path).exists():
            self._model_path = path
        conf = settings.value('yolo_detection/confidence', 0.5, type=float)
        self._confidence = conf

    def _ensure_model(self):
        if self._model is None:
            if not self._model_path:
                raise ValueError('No YOLO model selected')
            from ultralytics import YOLO
            import torch
            self._model = YOLO(self._model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model.to(device)

    def get_class_names(self) -> list[str] | None:
        try:
            self._ensure_model()
            return list(self._model.names.values())
        except Exception:
            return None

    def run(self, image_path: Path, **kwargs) -> list[dict]:
        self._ensure_model()
        from ltd.utils.file_utils import cv_imread
        img = cv_imread(image_path)
        if img is None:
            return []

        results = self._model(
            img,
            conf=self._confidence,
            verbose=False,
            stream=False,
        )

        detections = []
        for result in results:
            h, w = result.orig_shape
            task = self._model.task

            if task == 'obb' and result.obb is not None:
                for i, corners in enumerate(result.obb.xyxyxyxy):
                    cls_id = int(result.obb.cls[i].item())
                    conf = float(result.obb.conf[i].item())
                    cls_name = result.names[cls_id]

                    # Convert 4 corner points to normalized polygon
                    pts = corners.tolist()  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    polygon = [(pt[0] / w, pt[1] / h) for pt in pts]

                    # Infer bbox from polygon
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    cx = (min(xs) + max(xs)) / 2
                    cy = (min(ys) + max(ys)) / 2
                    bw = max(xs) - min(xs)
                    bh = max(ys) - min(ys)

                    detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': (cx, cy, bw, bh),
                        'polygon': polygon,
                        'mask': None,
                    })
            elif task == 'segment' and result.masks is not None:
                for i, mask_xy in enumerate(result.masks.xy):
                    cls_id = int(result.boxes.cls[i].item())
                    conf = float(result.boxes.conf[i].item())
                    cls_name = result.names[cls_id]

                    # Normalize polygon points
                    polygon = [(float(pt[0]) / w, float(pt[1]) / h)
                               for pt in mask_xy]

                    # Infer bbox
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    cx = (min(xs) + max(xs)) / 2
                    cy = (min(ys) + max(ys)) / 2
                    bw = max(xs) - min(xs)
                    bh = max(ys) - min(ys)

                    # Create binary mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    pts = mask_xy.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)

                    detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': (cx, cy, bw, bh),
                        'polygon': polygon,
                        'mask': mask,
                    })
            else:
                # Detection mode
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    cls_name = result.names[cls_id]

                    # Convert xyxy to normalized cxcywh
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': (cx, cy, bw, bh),
                        'polygon': None,
                        'mask': None,
                    })

        return detections

    def unload(self):
        self._model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
