"""Train YOLO tab: dataset selection, model config, training."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QVBoxLayout,
                               QWidget, QTextEdit)

from ltd.workers.training_worker import TrainingWorker

YOLO_DETECT_MODELS = {
    'YOLO26n': 'yolo26n.pt',
    'YOLO26s': 'yolo26s.pt',
    'YOLO26m': 'yolo26m.pt',
    'YOLO26l': 'yolo26l.pt',
    'YOLO26x': 'yolo26x.pt',
}

YOLO_SEG_MODELS = {
    'YOLO26n-seg': 'yolo26n-seg.pt',
    'YOLO26s-seg': 'yolo26s-seg.pt',
    'YOLO26m-seg': 'yolo26m-seg.pt',
    'YOLO26l-seg': 'yolo26l-seg.pt',
    'YOLO26x-seg': 'yolo26x-seg.pt',
}


class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainingWorker | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Settings group
        settings_group = QGroupBox('Training Settings')
        settings_layout = QVBoxLayout(settings_group)

        # Dataset path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel('Dataset path:'))
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText('Path to YOLO dataset folder')
        path_layout.addWidget(self.dataset_path_edit, stretch=1)
        self.browse_dataset_btn = QPushButton('Browse...')
        path_layout.addWidget(self.browse_dataset_btn)
        settings_layout.addLayout(path_layout)

        # Model type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel('Model type:'))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['Object Detection', 'Segmentation'])
        type_layout.addWidget(self.model_type_combo)
        type_layout.addStretch()
        settings_layout.addLayout(type_layout)

        # Base model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('Base model:'))
        self.base_model_combo = QComboBox()
        self._populate_models()
        model_layout.addWidget(self.base_model_combo, stretch=1)
        self.refresh_models_btn = QPushButton('Refresh')
        model_layout.addWidget(self.refresh_models_btn)
        settings_layout.addLayout(model_layout)

        # Model name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel('Model name:'))
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText('my_model')
        self.model_name_edit.setText('my_model')
        name_layout.addWidget(self.model_name_edit)
        settings_layout.addLayout(name_layout)

        # Train/Valid split
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel('Train/Valid split:'))
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(50, 95)
        self.split_slider.setValue(80)
        self.split_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.split_slider.setTickInterval(5)
        split_layout.addWidget(self.split_slider, stretch=1)
        self.split_label = QLabel('80%')
        self.split_label.setMinimumWidth(40)
        split_layout.addWidget(self.split_label)
        settings_layout.addLayout(split_layout)

        layout.addWidget(settings_group)

        # Start button
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton('Start Training')
        self.start_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.start_btn)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setVisible(False)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Status / Log
        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText('Training log will appear here...')
        layout.addWidget(self.log_output, stretch=1)

    def _connect_signals(self):
        self.browse_dataset_btn.clicked.connect(self._browse_dataset)
        self.refresh_models_btn.clicked.connect(self._populate_models)
        self.model_type_combo.currentIndexChanged.connect(self._populate_models)
        self.split_slider.valueChanged.connect(
            lambda v: self.split_label.setText(f'{v}%'))
        self.start_btn.clicked.connect(self._start_training)
        self.cancel_btn.clicked.connect(self._cancel_training)

    def _populate_models(self):
        self.base_model_combo.clear()

        # Pick detect vs segmentation preset models
        is_seg = self.model_type_combo.currentIndex() == 1
        preset = YOLO_SEG_MODELS if is_seg else YOLO_DETECT_MODELS
        for name, path in preset.items():
            self.base_model_combo.addItem(name, path)

        # Custom models from ./models/yolo/
        models_dir = Path('./models/yolo')
        if models_dir.exists():
            for pt_file in sorted(models_dir.glob('*.pt')):
                self.base_model_combo.addItem(
                    f'Custom: {pt_file.name}', str(pt_file))

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Dataset Folder')
        if folder:
            self.dataset_path_edit.setText(folder)

    def set_dataset_path(self, path: str):
        self.dataset_path_edit.setText(path)

    def set_model_type(self, model_type: str):
        """Set model type: 'detect' or 'segment'."""
        if model_type == 'segment':
            self.model_type_combo.setCurrentIndex(1)
        else:
            self.model_type_combo.setCurrentIndex(0)

    def _start_training(self):
        dataset_path = self.dataset_path_edit.text().strip()
        if not dataset_path:
            QMessageBox.warning(self, 'Warning', 'Please select a dataset path.')
            return

        if not Path(dataset_path).exists():
            QMessageBox.warning(self, 'Warning', 'Dataset path does not exist.')
            return

        model_name = self.model_name_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, 'Warning', 'Please enter a model name.')
            return

        model_path = self.base_model_combo.currentData()
        model_type = ('detect' if self.model_type_combo.currentIndex() == 0
                       else 'segment')
        split_ratio = self.split_slider.value()

        self._worker = TrainingWorker(
            dataset_path=dataset_path,
            model_path=model_path,
            model_name=model_name,
            model_type=model_type,
            split_ratio=split_ratio,
        )
        self._worker.status.connect(self._on_status)
        self._worker.error.connect(self._on_error)
        self._worker.training_complete.connect(self._on_training_complete)
        self._worker.finished_work.connect(self._on_finished)

        self.start_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.log_output.clear()
        self._worker.start()

    def _cancel_training(self):
        if self._worker:
            self._worker.cancel()

    def _on_status(self, text):
        self.status_label.setText(text)
        self.log_output.append(text)

    def _on_error(self, msg):
        self.log_output.append(f'ERROR: {msg}')
        QMessageBox.critical(self, 'Training Error', msg)

    def _on_training_complete(self, model_path):
        self.log_output.append(f'\nModel saved to: {model_path}')
        self._populate_models()  # Refresh model list

    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self._worker = None
        from ltd.utils.sound import play_completion_sound
        play_completion_sound()
