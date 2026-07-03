"""Minimal LM Studio client (OpenAI-compatible REST API).

LM Studio exposes an OpenAI-compatible server (default http://localhost:1234).
We use `/v1/models` to list loaded models and `/v1/chat/completions` with an
inline base64 image for vision captioning — no extra SDK dependency, same
`requests`-based style as `ltd/comfyui/client.py`.
"""
import base64
from pathlib import Path

import requests

from ltd.settings import DEFAULT_SETTINGS, get_settings

_MIME_BY_SUFFIX = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
}

# The model `type` values LM Studio's native API reports for vision-language
# (multimodal) models. Everything else ('llm', 'embeddings') is text-only.
_VISION_MODEL_TYPES = {'vlm'}


class LMStudioClient:
    """Client for a running LM Studio instance's OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = get_settings().value(
                'lmstudio_url', str(DEFAULT_SETTINGS['lmstudio_url']), type=str)
        self.base_url = (base_url or '').rstrip('/')

    def _root(self) -> str:
        # Host root without the OpenAI "/v1" suffix, for LM Studio's native API.
        base = self.base_url
        if base.endswith('/v1'):
            base = base[:-len('/v1')]
        return base.rstrip('/')

    def _api(self, path: str) -> str:
        # Accept either "http://host:port" or "http://host:port/v1".
        base = self.base_url
        if not base.endswith('/v1'):
            base = f'{base}/v1'
        return f'{base}{path}'

    def health_check(self) -> bool:
        try:
            r = requests.get(self._api('/models'), timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self, vision_only: bool = True) -> list[str]:
        """Return the ids of the models LM Studio exposes.

        Prefers LM Studio's native ``/api/v0/models`` endpoint, which reports a
        per-model ``type`` ('llm' | 'vlm' | 'embeddings') so vision-language
        models can be isolated. Falls back to the OpenAI-compatible
        ``/v1/models`` (no capability info, so returns everything) when the
        native endpoint isn't available (older LM Studio builds).
        """
        try:
            r = requests.get(f'{self._root()}/api/v0/models', timeout=10)
            r.raise_for_status()
            entries = r.json().get('data', [])
            if vision_only:
                entries = [m for m in entries
                           if (m.get('type') or '').lower()
                           in _VISION_MODEL_TYPES]
            return [m['id'] for m in entries if m.get('id')]
        except requests.RequestException:
            # Native API unavailable — fall back to the OpenAI-compatible list
            # (capabilities unknown there, so no vision filtering is possible).
            r = requests.get(self._api('/models'), timeout=10)
            r.raise_for_status()
            data = r.json()
            return [m['id'] for m in data.get('data', []) if m.get('id')]

    def unload_model(self, model_id: str) -> bool:
        """Best-effort unload of a loaded model to free VRAM/RAM.

        Uses LM Studio's v1 REST endpoint (``POST /api/v1/models/unload``,
        added in LM Studio 0.4.0). Returns False silently on older builds
        (404) or connection errors — unloading is a nicety, not critical.
        """
        if not model_id:
            return False
        try:
            r = requests.post(f'{self._root()}/api/v1/models/unload',
                              json={'instance_id': model_id}, timeout=30)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def caption(self, image_path: Path, model: str, system_prompt: str = '',
                user_text: str | None = None, timeout: int = 300) -> str:
        """Send one image (+ optional context) and return the model's text."""
        image_path = Path(image_path)
        mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), 'image/jpeg')
        b64 = base64.b64encode(image_path.read_bytes()).decode('ascii')

        content = [
            {'type': 'text', 'text': user_text or 'Describe this image.'},
            {'type': 'image_url',
             'image_url': {'url': f'data:{mime};base64,{b64}'}},
        ]
        messages = []
        if system_prompt.strip():
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': content})

        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': 4096,
            'stream': False,
        }
        r = requests.post(self._api('/chat/completions'),
                          json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        choices = data.get('choices') or []
        if not choices:
            raise ValueError('LM Studio returned no choices')
        return choices[0].get('message', {}).get('content', '') or ''
