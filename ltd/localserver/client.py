"""Client for a local OpenAI-compatible inference server ("Local server").

Captioning only needs the OpenAI-compatible ``POST /v1/chat/completions`` that
every popular local server exposes, so that part is identical everywhere.
Listing *vision* models and unloading a model after a batch are not part of the
OpenAI API, so the server's flavor is detected at runtime and its native
endpoints are used where they exist:

| Flavor         | Detected by                        | Vision filter                 | Unload                        |
|----------------|------------------------------------|-------------------------------|-------------------------------|
| Unsloth Studio | ``Server`` header / ``/api/health`` | active model via ``/v1/status`` | ``POST /v1/unload``           |
| LM Studio      | ``GET /api/v0/models``             | per-model ``type == 'vlm'``   | ``POST /api/v1/models/unload`` |
| Ollama         | ``GET /api/version``               | ``POST /api/show`` capabilities | ``keep_alive: 0``           |
| llama.cpp      | ``Server`` header / ``GET /props`` | ``modalities.vision``         | ``POST /models/unload`` (router) |
| vLLM           | ``GET /version``                   | none                          | none                          |
| anything else  | fallback                           | none                          | none                          |

`requests`-based, no SDK — same style as `ltd/comfyui/client.py`.
"""
import base64
import io
from pathlib import Path

import requests
from PIL import Image

from ltd.settings import DEFAULT_SETTINGS, get_settings

UNSLOTH = 'unsloth'
LMSTUDIO = 'lmstudio'
OLLAMA = 'ollama'
LLAMACPP = 'llamacpp'
VLLM = 'vllm'
OPENAI = 'openai'

FLAVOR_LABELS = {
    UNSLOTH: 'Unsloth Studio',
    LMSTUDIO: 'LM Studio',
    OLLAMA: 'Ollama',
    LLAMACPP: 'llama.cpp',
    VLLM: 'vLLM',
    OPENAI: 'OpenAI-compatible',
}

# Default megapixel budget for images sent to the model. Vision encoders resize
# or tile large images anyway; downscaling client-side cuts the upload and the
# vision-token count (ToriiGate / Qwen-VL were trained at ~1 MP).
DEFAULT_MAX_MEGAPIXELS = 1.0

# Formats every server accepts as an inline data URI. Anything else (bmp, tiff,
# gif, ...) is re-encoded before sending.
_PASSTHROUGH_MIME = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
}

# The model `type` values LM Studio's native API reports for vision-language
# models. Everything else ('llm', 'embeddings') is text-only.
_LMSTUDIO_VISION_TYPES = {'vlm'}

_PROBE_TIMEOUT = 5

# (root URL, API key) -> flavor. Detection costs a few requests and the server
# behind a URL rarely changes; detect_flavor(refresh=True) re-probes.
_flavor_cache: dict[tuple[str, str], str] = {}


def encode_image_data_uri(image_path: Path,
                          max_megapixels: float = DEFAULT_MAX_MEGAPIXELS) -> str:
    """Return the image as a base64 ``data:`` URI within a megapixel budget.

    ``max_megapixels <= 0`` disables downscaling. An image already within the
    budget in a widely supported format is sent byte-for-byte; anything else
    is re-encoded as JPEG, or PNG when it has transparency.
    """
    image_path = Path(image_path)
    mime = _PASSTHROUGH_MIME.get(image_path.suffix.lower())
    with Image.open(image_path) as img:
        pixels = img.width * img.height
        budget = max_megapixels * 1_000_000
        oversized = max_megapixels > 0 and pixels > budget
        if mime and not oversized:
            data = image_path.read_bytes()
        else:
            has_alpha = (img.mode in ('RGBA', 'LA', 'PA')
                         or 'transparency' in img.info)
            # Convert before resizing: Pillow resizes palette images with
            # nearest-neighbour regardless of the filter asked for.
            img = img.convert('RGBA' if has_alpha else 'RGB')
            if oversized:
                scale = (budget / pixels) ** 0.5
                img = img.resize((max(int(img.width * scale), 1),
                                  max(int(img.height * scale), 1)),
                                 Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            if has_alpha:
                img.save(buffer, format='PNG')
                mime = 'image/png'
            else:
                img.save(buffer, format='JPEG', quality=95)
                mime = 'image/jpeg'
            data = buffer.getvalue()
    return f'data:{mime};base64,{base64.b64encode(data).decode("ascii")}'


def _json_object(r: requests.Response) -> dict:
    """The response body as a dict, or {} when it isn't a JSON object."""
    try:
        body = r.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _error_detail(r: requests.Response) -> str:
    """Best human-readable error message from a failed response."""
    body = _json_object(r)
    if not body:
        return (r.text or r.reason or '').strip()[:300]
    # OpenAI style {"error": {"message": ...}}, FastAPI style {"detail": ...}
    err = body.get('error', body.get('detail', body))
    if isinstance(err, dict):
        err = err.get('message', err)
    return str(err)[:300]


class LocalServerClient:
    """Client for the server at the ``local_server_url`` setting."""

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None):
        settings = get_settings()
        default_url = str(DEFAULT_SETTINGS['local_server_url'])
        if base_url is None:
            base_url = settings.value('local_server_url', default_url, type=str)
        if api_key is None:
            api_key = settings.value('local_server_api_key', '', type=str)
        self.base_url = (base_url or default_url).strip().rstrip('/')
        self.api_key = (api_key or '').strip()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _root(self) -> str:
        # Accept either "http://host:port" or "http://host:port/v1".
        base = self.base_url
        if base.endswith('/v1'):
            base = base[:-len('/v1')]
        return base.rstrip('/')

    def _api(self, path: str) -> str:
        return f'{self._root()}/v1{path}'

    def _headers(self) -> dict:
        if not self.api_key:
            return {}
        return {'Authorization': f'Bearer {self.api_key}'}

    def _get(self, url: str, timeout: float = 10) -> requests.Response:
        return requests.get(url, headers=self._headers(), timeout=timeout)

    def _post(self, url: str, payload: dict,
              timeout: float = 30) -> requests.Response:
        return requests.post(url, headers=self._headers(), json=payload,
                             timeout=timeout)

    def _check(self, r: requests.Response) -> requests.Response:
        """raise_for_status, carrying the server's message and an API-key hint."""
        if r.ok:
            return r
        detail = _error_detail(r)
        if r.status_code in (401, 403):
            # Hint first: status lines elide the middle of long messages.
            hint = ('API key rejected' if self.api_key else
                    'API key required, set it in the Settings tab')
            raise PermissionError(f'{hint} (HTTP {r.status_code}: {detail})')
        raise requests.HTTPError(f'HTTP {r.status_code}: {detail}', response=r)

    def _probe(self, url: str, payload: dict | None = None) -> dict:
        """GET (or POST ``payload``) and return the JSON object, {} on any failure."""
        try:
            if payload is None:
                r = self._get(url, timeout=_PROBE_TIMEOUT)
            else:
                r = self._post(url, payload, timeout=_PROBE_TIMEOUT)
        except requests.RequestException:
            return {}
        return _json_object(r) if r.ok else {}

    # ------------------------------------------------------------------
    # Flavor detection
    # ------------------------------------------------------------------

    def detect_flavor(self, refresh: bool = False) -> str:
        """Identify which server app is behind the URL (one of the flavor constants)."""
        key = (self._root(), self.api_key)
        if not refresh and key in _flavor_cache:
            return _flavor_cache[key]
        flavor = self._probe_flavor()
        _flavor_cache[key] = flavor
        return flavor

    def _probe_flavor(self) -> str:
        root = self._root()
        # The first request doubles as the reachability check: failing to
        # connect means there is no server, not that its flavor is unknown.
        try:
            r = self._get(f'{root}/api/health', timeout=_PROBE_TIMEOUT)
        except requests.RequestException as e:
            raise ConnectionError(
                f'Cannot reach local server at {root}: {e}') from e
        server = r.headers.get('Server', '').lower()
        service = str(_json_object(r).get('service', '')).lower()
        if 'unsloth' in server or 'unsloth' in service:
            return UNSLOTH
        if 'llama.cpp' in server:
            return LLAMACPP

        # Native endpoints unique to each app, most specific first.
        if 'data' in self._probe(f'{root}/api/v0/models'):
            return LMSTUDIO
        if 'version' in self._probe(f'{root}/api/version'):
            return OLLAMA
        props = self._probe(f'{root}/props')
        if 'default_generation_settings' in props or 'modalities' in props:
            return LLAMACPP
        if 'version' in self._probe(f'{root}/version'):
            return VLLM
        return OPENAI

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def list_models(self, vision_only: bool = True) -> tuple[list[str], bool]:
        """Return ``(model_ids, vision_filtered)``.

        ``vision_filtered`` is True when the server reported capabilities and
        the list was narrowed to vision models; False means capabilities are
        unknown and the list may include text-only models.
        """
        listers = {
            UNSLOTH: self._list_unsloth,
            LMSTUDIO: self._list_lmstudio,
            OLLAMA: self._list_ollama,
            LLAMACPP: self._list_llamacpp,
        }
        lister = listers.get(self.detect_flavor())
        if lister is None:
            return [m['id'] for m in self._openai_models()], False
        return lister(vision_only)

    def _openai_models(self) -> list[dict]:
        r = self._check(self._get(self._api('/models')))
        return [m for m in _json_object(r).get('data', [])
                if isinstance(m, dict) and m.get('id')]

    def _list_unsloth(self, vision_only: bool) -> tuple[list[str], bool]:
        # Unsloth lists every downloaded model with a `loaded` flag, not just
        # what is in memory — put the loaded one(s) first.
        entries = self._openai_models()
        loaded = [m['id'] for m in entries if m.get('loaded')]
        ids = loaded + [m['id'] for m in entries if not m.get('loaded')]
        if vision_only and loaded:
            # Capabilities are only reported for the active model.
            if self._probe(self._api('/status')).get('is_vision') is False:
                ids = [i for i in ids if i not in loaded]
        return ids, False

    def _list_lmstudio(self, vision_only: bool) -> tuple[list[str], bool]:
        r = self._check(self._get(f'{self._root()}/api/v0/models'))
        entries = [m for m in _json_object(r).get('data', []) if m.get('id')]
        if vision_only:
            entries = [m for m in entries
                       if (m.get('type') or '').lower()
                       in _LMSTUDIO_VISION_TYPES]
        return [m['id'] for m in entries], vision_only

    def _list_ollama(self, vision_only: bool) -> tuple[list[str], bool]:
        root = self._root()
        r = self._check(self._get(f'{root}/api/tags'))
        names = [m.get('model') or m.get('name')
                 for m in _json_object(r).get('models', [])]
        names = [n for n in names if n]
        if not vision_only:
            return names, False
        vision, filtered = [], False
        for name in names:
            caps = self._probe(f'{root}/api/show', {'model': name}) \
                .get('capabilities')
            if caps is None:
                vision.append(name)  # older Ollama: unknown, keep it
            else:
                filtered = True
                if 'vision' in caps:
                    vision.append(name)
        return vision, filtered

    def _list_llamacpp(self, vision_only: bool) -> tuple[list[str], bool]:
        ids = [m['id'] for m in self._openai_models()]
        modalities = self._probe(f'{self._root()}/props').get('modalities')
        vision = modalities.get('vision') if isinstance(modalities, dict) \
            else None
        if vision_only and vision is False:
            return [], True
        return ids, vision is True

    def unload_model(self, model_id: str) -> bool:
        """Best-effort unload to free VRAM/RAM.

        Returns False when the flavor has no unload endpoint, or on any error —
        unloading is a nicety, never worth failing a batch over.
        """
        if not model_id:
            return False
        root = self._root()
        endpoints = {
            UNSLOTH: (f'{root}/v1/unload', {'model_path': model_id}),
            LMSTUDIO: (f'{root}/api/v1/models/unload',
                       {'instance_id': model_id}),
            OLLAMA: (f'{root}/api/generate',
                     {'model': model_id, 'keep_alive': 0}),
            LLAMACPP: (f'{root}/models/unload', {'model': model_id}),
        }
        try:
            endpoint = endpoints.get(self.detect_flavor())
            if endpoint is None:
                return False
            url, payload = endpoint
            return self._post(url, payload, timeout=120).ok
        except OSError:  # requests errors and our ConnectionError
            return False

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def caption(self, image_path: Path, model: str, system_prompt: str = '',
                user_text: str | None = None,
                max_megapixels: float = DEFAULT_MAX_MEGAPIXELS,
                max_tokens: int = 4096, timeout: int = 300) -> str:
        """Send one image (+ optional context) and return the model's text."""
        content = [
            {'type': 'text', 'text': user_text or 'Describe this image.'},
            {'type': 'image_url',
             'image_url': {'url': encode_image_data_uri(image_path,
                                                        max_megapixels)}},
        ]
        messages = []
        if system_prompt.strip():
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': content})

        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'stream': False,
        }
        r = self._check(self._post(self._api('/chat/completions'), payload,
                                   timeout=timeout))
        choices = _json_object(r).get('choices') or []
        if not choices:
            raise ValueError('Local server returned no choices')
        return choices[0].get('message', {}).get('content', '') or ''
