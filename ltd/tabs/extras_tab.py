import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ltd.settings import get_settings
from ltd.widgets.collapsible_panel import CollapsiblePanel

logger = logging.getLogger(__name__)

EXTRAS_DIR = Path(__file__).resolve().parents[2] / 'extras_scripts'

REQUIRED_PARAM_KEYS = {'name', 'type', 'label', 'default'}
VALID_PARAM_TYPES = {'str', 'bool', 'folder'}


def _validate_script_info(info: dict) -> bool:
    """Check SCRIPT_INFO has required structure."""
    if not isinstance(info, dict):
        return False
    if 'name' not in info or 'parameters' not in info:
        return False
    if not isinstance(info['parameters'], list):
        return False
    for p in info['parameters']:
        if not isinstance(p, dict):
            return False
        if not REQUIRED_PARAM_KEYS.issubset(p.keys()):
            return False
        if p['type'] not in VALID_PARAM_TYPES:
            return False
    return True


def _load_script(path: Path):
    """Import a script module from path, validate, and return it or None."""
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        logger.warning('Failed to load extras script %s: %s', path.name, e)
        return None

    info = getattr(mod, 'SCRIPT_INFO', None)
    if info is None or not _validate_script_info(info):
        logger.warning('Invalid SCRIPT_INFO in %s', path.name)
        return None

    if not callable(getattr(mod, 'check_available', None)):
        logger.warning('Missing check_available() in %s', path.name)
        return None
    if not callable(getattr(mod, 'build_command', None)):
        logger.warning('Missing build_command() in %s', path.name)
        return None

    return mod


class ExtrasTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scripts: list[dict] = []
        self._settings = get_settings()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Scroll area for panels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._container)

        self._discover_scripts()

    def _discover_scripts(self):
        """Scan extras_scripts/ and build panels."""
        if not EXTRAS_DIR.is_dir():
            return

        for py_file in sorted(EXTRAS_DIR.glob('*.py')):
            if py_file.name.startswith('_'):
                continue
            mod = _load_script(py_file)
            if mod is None:
                continue
            self._add_script_panel(mod)

    def _settings_key(self, script_stem: str, param_name: str) -> str:
        return f'extras/{script_stem}/{param_name}'

    def _add_script_panel(self, mod):
        info = mod.SCRIPT_INFO
        script_stem = Path(mod.__spec__.origin).stem
        panel = CollapsiblePanel(info['name'], self)

        # Check availability
        try:
            available, reason = mod.check_available()
        except Exception as e:
            available, reason = False, str(e)

        if not available:
            panel.set_available(False)
            panel.set_status_text(reason)

        # Build parameter widgets
        param_widgets = {}
        for param in info['parameters']:
            ptype = param['type']
            name = param['name']
            key = self._settings_key(script_stem, name)

            if ptype == 'bool':
                cb = QCheckBox(param['label'])
                saved = self._settings.value(key)
                if saved is not None:
                    cb.setChecked(saved == 'true')
                else:
                    cb.setChecked(bool(param['default']))
                cb.toggled.connect(lambda val, _k=key: self._settings.setValue(_k, str(val).lower()))
                panel.content_layout.addWidget(cb)
                param_widgets[name] = cb

            elif ptype == 'folder':
                lbl = QLabel(param['label'])
                panel.content_layout.addWidget(lbl)
                row = QHBoxLayout()
                saved = self._settings.value(key)
                le = QLineEdit(saved if saved is not None else str(param['default']))
                le.setPlaceholderText(param.get('placeholder', ''))
                le.textChanged.connect(lambda val, _k=key: self._settings.setValue(_k, val))
                btn = QPushButton('Browse...')
                btn.clicked.connect(lambda checked=False, _le=le: self._browse_folder(_le))
                row.addWidget(le)
                row.addWidget(btn)
                panel.content_layout.addLayout(row)
                param_widgets[name] = le

            else:  # str
                lbl = QLabel(param['label'])
                panel.content_layout.addWidget(lbl)
                saved = self._settings.value(key)
                le = QLineEdit(saved if saved is not None else str(param['default']))
                le.setPlaceholderText(param.get('placeholder', ''))
                le.textChanged.connect(lambda val, _k=key: self._settings.setValue(_k, val))
                panel.content_layout.addWidget(le)
                param_widgets[name] = le

        # Description
        if info.get('description'):
            desc_label = QLabel(info['description'])
            desc_label.setStyleSheet('color: gray; font-size: 11px;')
            desc_label.setWordWrap(True)
            panel.content_layout.addWidget(desc_label)

        # Execute button
        exec_btn = QPushButton('Execute')
        exec_btn.clicked.connect(
            lambda checked=False, _mod=mod, _pw=param_widgets, _params=info['parameters']:
                self._execute(_mod, _pw, _params)
        )
        panel.content_layout.addWidget(exec_btn)

        self._container_layout.addWidget(panel)
        self._scripts.append({'mod': mod, 'panel': panel, 'widgets': param_widgets})

    def _browse_folder(self, line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder', line_edit.text())
        if folder:
            line_edit.setText(folder)

    def _execute(self, mod, param_widgets: dict, params_info: list):
        """Collect parameter values, build command, spawn in new terminal."""
        values = {}
        for p in params_info:
            name = p['name']
            widget = param_widgets[name]
            if p['type'] == 'bool':
                values[name] = widget.isChecked()
            else:
                values[name] = widget.text()

        try:
            cmd = mod.build_command(values)
        except Exception as e:
            logger.error('build_command failed: %s', e)
            return

        logger.info('Extras: executing %s', cmd)

        try:
            if sys.platform == 'win32':
                if isinstance(cmd, str):
                    # Raw command string (batch syntax) — pass through shell
                    full = f'start cmd /k {cmd}'
                    subprocess.Popen(full, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen(
                        ['cmd', '/c', 'start', 'cmd', '/k'] + cmd,
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
            else:
                # Linux/macOS fallback
                subprocess.Popen(cmd, shell=isinstance(cmd, str))
        except Exception as e:
            logger.error('Failed to spawn command: %s', e)
