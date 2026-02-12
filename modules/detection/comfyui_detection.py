"""Built-in ComfyUI detection module."""
import copy
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from modules.base import BaseDetectionModule
from ltd.comfyui.client import ComfyUIClient
from ltd.comfyui.workflow import (load_workflow, validate_detection_workflow,
                                   set_input_image, find_nodes_by_title,
                                   LTD_OUTPUT_MASK, LTD_OUTPUT_IMAGE)
from ltd.utils.file_utils import get_temp_dir_no_clear
from ltd.widgets.workflow_selector import WorkflowSelector


class ComfyUIDetectionModule(BaseDetectionModule):

    def __init__(self):
        self._settings_widget = None
        self._workflow_selector = None
        self._workflow_text = ''
        self._prepared_workflow = None
        self._client = None

    @property
    def name(self) -> str:
        return 'ComfyUI Workflow'

    def get_settings_widget(self) -> QWidget:
        if self._settings_widget is not None:
            return self._settings_widget

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel('ComfyUI workflow:'))
        self._workflow_selector = WorkflowSelector(settings_key='detection')
        layout.addWidget(self._workflow_selector)

        info = QLabel('Required nodes: LTD_Input_Image, '
                       'LTD_Output_Mask or LTD_Output_Image')
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        self._settings_widget = widget
        return widget

    def prepare(self):
        """Read widget state and pre-parse workflow + client (main thread)."""
        if self._workflow_selector:
            self._workflow_text = self._workflow_selector.get_workflow_text()

        self._prepared_workflow = None
        self._client = None

        if not self._workflow_text:
            raise ValueError('No workflow selected')

        workflow = load_workflow(self._workflow_text)
        if workflow is None:
            raise ValueError('Invalid workflow JSON')

        valid, msg = validate_detection_workflow(workflow)
        if not valid:
            raise ValueError(f'Workflow validation failed: {msg}')

        self._prepared_workflow = workflow
        self._client = ComfyUIClient()
        if not self._client.health_check():
            self._client = None
            raise ConnectionError('Cannot connect to ComfyUI')

    def run(self, image_path: Path, **kwargs) -> list[dict]:
        if not self._prepared_workflow:
            raise ValueError('No workflow provided (call prepare() first)')
        if not self._client:
            raise ConnectionError('ComfyUI client not ready')

        # Deep copy so set_input_image doesn't mutate the template
        workflow = copy.deepcopy(self._prepared_workflow)

        # Upload image
        upload_result = self._client.upload_image(image_path)
        image_filename = upload_result.get('name', image_path.name)
        set_input_image(workflow, image_filename)

        # Run
        output_dir = get_temp_dir_no_clear('comfyui_detection_output')
        result = self._client.run_workflow(workflow, output_dir)

        detections = []
        # Process output masks
        import cv2
        for output_file in result['files']:
            mask = cv2.imread(str(output_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Threshold
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                # Find contours for bbox
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                h, w = mask.shape
                for contour in contours:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    cx = (x + cw / 2) / w
                    cy = (y + ch / 2) / h
                    detections.append({
                        'class_id': 0,
                        'class_name': 'detected',
                        'confidence': 1.0,
                        'bbox': (cx, cy, cw / w, ch / h),
                        'polygon': None,
                        'mask': mask,
                    })
                    break  # One detection per mask file

        return detections
