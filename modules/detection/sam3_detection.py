"""Built-in SAM3 text-prompt (open-vocabulary) detection module."""
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from modules.base import BaseDetectionModule
from modules.sam3.engine import get_shared_engine


class Sam3DetectionModule(BaseDetectionModule):

    def __init__(self):
        self._model_path = ''
        self._confidence = 0.2
        self._phrases: list[str] = []
        self._settings_widget = None
        self._model_path_label = None
        self._conf_label = None
        self._prompt_edit = None
        self._restore_settings()

    @property
    def name(self) -> str:
        return 'SAM3 (text prompt)'

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
                      else 'Auto-download')
        self._model_path_label = QLabel(label_text)
        self._model_path_label.setWordWrap(True)
        path_layout.addWidget(self._model_path_label, stretch=1)
        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Text prompt — one phrase per line
        layout.addWidget(QLabel('Text prompt (one phrase per line):'))
        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setPlainText('\n'.join(self._phrases))
        self._prompt_edit.setPlaceholderText('cat\ndog\nred car')
        self._prompt_edit.setFixedHeight(80)
        layout.addWidget(self._prompt_edit)
        info = QLabel('Each non-empty line is queried independently and '
                       'becomes its own class (matched by name against your '
                       'Classes list).')
        info.setWordWrap(True)
        layout.addWidget(info)

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
            None, 'Select SAM3 Model', './models/sam3',
            'Safetensors (*.safetensors);;All Files (*)')
        if path:
            self._model_path = path
            self._model_path_label.setText(Path(path).name)
            get_shared_engine().set_model_path(self._model_path)
            self._save_settings()

    def _on_conf_slider_changed(self, value):
        self._confidence = value / 100.0
        if self._conf_label:
            self._conf_label.setText(f'{self._confidence:.2f}')
        self._save_settings()

    def _save_settings(self):
        from ltd.settings import get_settings
        settings = get_settings()
        settings.setValue('sam3/model_path', self._model_path)
        settings.setValue('sam3_detection/confidence', self._confidence)

    def _restore_settings(self):
        from ltd.settings import get_settings
        settings = get_settings()
        path = settings.value('sam3/model_path', '', type=str)
        if path and Path(path).exists():
            self._model_path = path
        self._confidence = settings.value(
            'sam3_detection/confidence', 0.2, type=float)

    def prepare(self):
        """Read widget state (GUI thread only) into plain attributes."""
        if self._prompt_edit is not None:
            self._phrases = [
                line.strip() for line in self._prompt_edit.toPlainText().splitlines()
                if line.strip()
            ]
        if not self._phrases:
            raise ValueError('SAM3 detection requires at least one text prompt line')

    def get_class_names(self) -> list[str] | None:
        return self._phrases or None

    def run(self, image_path: Path, **kwargs) -> list[dict]:
        if not self._phrases:
            return []
        from ltd.utils.file_utils import cv_imread
        from ltd.utils.mask_utils import mask_to_bbox_normalized, mask_to_polygons

        image = cv_imread(image_path)
        if image is None:
            return []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        engine = get_shared_engine()
        engine.set_model_path(self._model_path or None)
        results = engine.segment_text(image_rgb, self._phrases, self._confidence)

        detections = []
        for det in results:
            mask = det['mask']
            polygons = mask_to_polygons(mask)
            bbox = mask_to_bbox_normalized(mask)
            if not polygons or bbox is None:
                continue
            class_id = self._phrases.index(det['phrase'])
            for polygon in polygons:
                detections.append({
                    'class_id': class_id,
                    'class_name': det['phrase'],
                    'confidence': det['score'],
                    'bbox': bbox,
                    'polygon': polygon,
                    'mask': mask,
                })
        return detections

    def unload(self):
        # The shared engine may still be needed by Magic Wand in this
        # session, so it isn't torn down here — only GPU scratch memory
        # from this run is released (mirrors BaseWorker._cleanup_memory).
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
