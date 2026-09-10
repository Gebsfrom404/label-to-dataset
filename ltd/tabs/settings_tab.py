"""Settings tab: server connections and appearance (formerly the top toolbar)."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout, QWidget)

from ltd.settings import DEFAULT_SETTINGS
from ltd.widgets.settings_widgets import (SettingsComboBox, SettingsLineEdit,
                                           SettingsSpinBox)


class SettingsTab(QWidget):
    """App-wide settings. Every field persists to QSettings as it is edited."""
    theme_changed = Signal(str)
    font_size_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        # Fill the tab up to a readable width, so fields don't stretch across
        # very wide windows. Added *without* an alignment flag on purpose: an
        # aligned layout item is pinned to its size hint, and word-wrapped
        # labels keep that hint narrow — the whole column collapsed to ~450px.
        content = QWidget()
        content.setMaximumWidth(900)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_comfyui_group())
        layout.addWidget(self._build_local_server_group())
        layout.addWidget(self._build_appearance_group())
        outer.addWidget(content)
        outer.addStretch()

    # Word-wrapped labels go in each group's QVBoxLayout, not in QFormLayout
    # rows: a spanning form row (addRow(widget)) didn't grow to the wrapped
    # text's height and clipped it, squeezing the rows below with it.
    @staticmethod
    def _wrapped_label(text: str = '') -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _set_status(label: QLabel, text: str):
        """Show a status message; an empty one hides the label (no blank line)."""
        label.setText(text)
        label.setVisible(bool(text))

    def _build_comfyui_group(self) -> QGroupBox:
        group = QGroupBox('ComfyUI')
        box = QVBoxLayout(group)
        form = QFormLayout()

        self.comfyui_url = SettingsLineEdit(
            'comfyui_url', DEFAULT_SETTINGS['comfyui_url'])
        self.comfyui_url.setPlaceholderText(DEFAULT_SETTINGS['comfyui_url'])
        self.comfyui_test_btn = QPushButton('Test')
        self.comfyui_test_btn.clicked.connect(self._test_comfyui)
        row = QHBoxLayout()
        row.addWidget(self.comfyui_url, 1)
        row.addWidget(self.comfyui_test_btn)
        form.addRow('URL:', row)
        box.addLayout(form)

        self.comfyui_status = self._wrapped_label()
        self.comfyui_status.setVisible(False)
        box.addWidget(self.comfyui_status)
        return group

    def _build_local_server_group(self) -> QGroupBox:
        group = QGroupBox('Local server')
        box = QVBoxLayout(group)

        box.addWidget(self._wrapped_label(
            'Vision-model captioning in the Caption tab. Any OpenAI-compatible '
            'server works; Unsloth Studio, LM Studio, Ollama, llama.cpp and '
            'vLLM are detected automatically for model filtering and '
            'unloading.'))

        form = QFormLayout()
        self.local_server_url = SettingsLineEdit(
            'local_server_url', DEFAULT_SETTINGS['local_server_url'])
        self.local_server_url.setPlaceholderText(
            DEFAULT_SETTINGS['local_server_url'])
        self.local_server_url.setToolTip(
            'Server address, with or without a trailing "/v1"')
        self.local_server_test_btn = QPushButton('Test')
        self.local_server_test_btn.clicked.connect(self._test_local_server)
        url_row = QHBoxLayout()
        url_row.addWidget(self.local_server_url, 1)
        url_row.addWidget(self.local_server_test_btn)
        form.addRow('URL:', url_row)

        self.local_server_api_key = SettingsLineEdit(
            'local_server_api_key', DEFAULT_SETTINGS['local_server_api_key'])
        self.local_server_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.local_server_api_key.setPlaceholderText(
            'Optional; sent as a Bearer token')
        self.local_server_api_key.setToolTip(
            'Needed when the server enforces authentication, e.g. Unsloth '
            'Studio without keyless API access, or llama.cpp started with '
            '--api-key. Stored in the app settings in plain text.')
        show_key = QCheckBox('Show')
        show_key.toggled.connect(
            lambda on: self.local_server_api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if on
                else QLineEdit.EchoMode.Password))
        key_row = QHBoxLayout()
        key_row.addWidget(self.local_server_api_key, 1)
        key_row.addWidget(show_key)
        form.addRow('API key:', key_row)
        box.addLayout(form)

        self.local_server_status = self._wrapped_label()
        self.local_server_status.setVisible(False)
        box.addWidget(self.local_server_status)
        return group

    def _build_appearance_group(self) -> QGroupBox:
        group = QGroupBox('Appearance')
        form = QFormLayout(group)

        self.theme_combo = SettingsComboBox('theme', DEFAULT_SETTINGS['theme'])
        self.theme_combo.addItems(['dark', 'light'])
        self.theme_combo.currentTextChanged.connect(self.theme_changed.emit)
        form.addRow('Theme:', self.theme_combo)

        self.font_size_spin = SettingsSpinBox(
            'font_size', DEFAULT_SETTINGS['font_size'], 8, 24)
        self.font_size_spin.setSuffix('pt')
        self.font_size_spin.valueChanged.connect(self.font_size_changed.emit)
        form.addRow('Font size:', self.font_size_spin)
        return group

    def _test_comfyui(self):
        from ltd.comfyui.client import ComfyUIClient
        self._set_status(self.comfyui_status, 'Connecting...')
        QApplication.processEvents()
        if ComfyUIClient().health_check():
            self._set_status(self.comfyui_status, 'Connected')
        else:
            self._set_status(self.comfyui_status, 'Cannot connect to ComfyUI')

    def _test_local_server(self):
        """Detect the server flavor and count its models."""
        from ltd.localserver.client import FLAVOR_LABELS, LocalServerClient
        status = self.local_server_status
        self._set_status(status, 'Connecting...')
        QApplication.processEvents()
        client = LocalServerClient()
        try:
            label = FLAVOR_LABELS[client.detect_flavor(refresh=True)]
        except Exception as e:
            self._set_status(status, str(e))
            return
        try:
            models, vision_filtered = client.list_models()
        except Exception as e:
            self._set_status(status, f'Detected {label}, but: {e}')
            return
        kind = 'vision model(s)' if vision_filtered else 'model(s)'
        self._set_status(
            status, f'Connected to {label}: {len(models)} {kind} available')
