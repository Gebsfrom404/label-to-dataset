"""Built-in ComfyUI modification module."""
import copy
from pathlib import Path

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from modules.base import BaseModificationModule
from ltd.comfyui.client import ComfyUIClient
from ltd.comfyui.workflow import (load_workflow, validate_modification_workflow,
                                   set_input_image, set_input_mask)
from ltd.utils.file_utils import get_temp_dir_no_clear
from ltd.widgets.workflow_selector import WorkflowSelector


class ComfyUIModificationModule(BaseModificationModule):

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
        self._workflow_selector = WorkflowSelector(settings_key='modification')
        layout.addWidget(self._workflow_selector)

        info = QLabel('Required nodes: LTD_Input_Image, '
                       'LTD_Input_Mask, LTD_Output_Image')
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

        valid, msg = validate_modification_workflow(workflow)
        if not valid:
            raise ValueError(f'Workflow validation failed: {msg}')

        self._prepared_workflow = workflow
        self._client = ComfyUIClient()
        if not self._client.health_check():
            self._client = None
            raise ConnectionError('Cannot connect to ComfyUI')

    def run(self, image_path: Path, mask_path: Path, **kwargs) -> Path:
        if not self._prepared_workflow:
            raise ValueError('No workflow provided (call prepare() first)')
        if not self._client:
            raise ConnectionError('ComfyUI client not ready')

        # Deep copy so set_input_* doesn't mutate the template
        workflow = copy.deepcopy(self._prepared_workflow)

        # Upload image and mask
        img_result = self._client.upload_image(image_path)
        mask_result = self._client.upload_image(mask_path)
        set_input_image(workflow, img_result.get('name', image_path.name))
        set_input_mask(workflow, mask_result.get('name', mask_path.name))

        # Run
        output_dir = get_temp_dir_no_clear('comfyui_modification_output')
        result = self._client.run_workflow(workflow, output_dir)

        if result['files']:
            # Rename to unique name based on input image to avoid collisions
            output_file = result['files'][0]
            final_path = output_dir / f'{image_path.stem}_modified{output_file.suffix}'
            if final_path.exists():
                final_path.unlink()
            output_file.rename(final_path)
            return final_path
        raise RuntimeError('ComfyUI workflow produced no output image')
