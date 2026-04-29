"""Parsers for generation metadata embedded in PNG files (forge / ComfyUI)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypeGuard


_LORA_RE = re.compile(r'<lora:[^<>]*?>', re.IGNORECASE)
# Region-control / scheduling keywords that should split a prompt the same
# way a comma does. Whole-word, uppercase only — these are reserved tokens
# in a1111 / comfy graphs, not naturally-occurring tag text.
_PROMPT_KEYWORD_RE = re.compile(
    r'\b(?:AND|BREAK|ADDBASE|ADDCOMM|ADDCOL|ADDROW)\b')


def split_prompt_tags(prompt: str) -> list[str]:
    """Split a prompt string into tags.

    Treats commas as the primary separator, additionally:
      - any ``<lora:...>`` block is forced into its own tag (commas are
        injected on both sides before splitting), so it stays standalone
        even when neighbouring tokens omit commas
      - the keywords AND, BREAK, ADDBASE, ADDCOMM, ADDCOL, ADDROW are
        treated as commas (they vanish, but split the surrounding text)
    """
    if not prompt:
        return []
    text = _LORA_RE.sub(lambda m: f',{m.group(0)},', prompt)
    text = _PROMPT_KEYWORD_RE.sub(',', text)
    return [t.strip() for t in text.split(',') if t.strip()]


@dataclass
class GenMetadata:
    format: str = ''  # 'forge' | 'comfyui' | ''
    positive_prompt: str = ''
    negative_prompt: str = ''
    settings: dict = field(default_factory=dict)
    width: int = 0
    height: int = 0
    raw_text: str = ''
    workflow: Optional[dict] = None

    @property
    def tags(self) -> list[str]:
        return split_prompt_tags(self.positive_prompt)

    @property
    def resolution_str(self) -> str:
        if self.width and self.height:
            return f'{self.width}x{self.height}'
        return ''

    @property
    def has_data(self) -> bool:
        return bool(self.format)


# ---------------------------------------------------------------------------
# PNG text chunk reader
# ---------------------------------------------------------------------------

def read_png_text(path: Path) -> dict[str, str]:
    """Return {key: value} for textual PNG chunks (tEXt / iTXt / zTXt)."""
    try:
        from PIL import Image
        with Image.open(str(path)) as img:
            return {str(k): v for k, v in img.info.items()
                    if isinstance(v, str)}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Forge / Auto1111 parser
# ---------------------------------------------------------------------------

_FORGE_NEG_RE = re.compile(r'(?:^|\n)Negative prompt:\s*')
_FORGE_SETTINGS_LINE_RE = re.compile(r'(?:^|\n)(Steps:\s*\d)')
# Splits k:v pairs on the settings line. Match next "Key: " boundary
# (allows spaces in keys, e.g. "Schedule type:", "Model hash:").
_FORGE_KV_BOUNDARY_RE = re.compile(r',\s+([A-Za-z][A-Za-z0-9_ %]*?):\s+')


def _parse_forge_settings(text: str) -> dict:
    """Parse a single-line ", "-separated key:value settings string.

    Handles values that contain commas when they are quoted ("...") or
    enclosed in JSON braces ({...}).
    """
    out: dict[str, str] = {}
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in ' ,':
            i += 1
        if i >= n:
            break
        key_end = text.find(':', i)
        if key_end < 0:
            break
        key = text[i:key_end].strip()
        i = key_end + 1
        while i < n and text[i] == ' ':
            i += 1
        if i < n and text[i] == '"':
            end = text.find('"', i + 1)
            if end < 0:
                end = n
            value = text[i + 1:end]
            i = end + 1
        elif i < n and text[i] == '{':
            depth, start = 0, i
            while i < n:
                c = text[i]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            value = text[start:i]
        else:
            m = _FORGE_KV_BOUNDARY_RE.search(text, i)
            if m:
                value = text[i:m.start()]
                i = m.start()
            else:
                value = text[i:]
                i = n
            value = value.rstrip(', ').strip()
        if key:
            out[key] = value
    return out


def parse_forge(text: str) -> GenMetadata:
    md = GenMetadata(format='forge', raw_text=text)

    matches = list(_FORGE_SETTINGS_LINE_RE.finditer(text))
    if matches:
        m = matches[-1]
        before = text[:m.start(1)].rstrip('\n')
        settings_text = text[m.start(1):].strip()
    else:
        before = text
        settings_text = ''

    neg = _FORGE_NEG_RE.search(before)
    if neg:
        md.positive_prompt = before[:neg.start()].strip()
        md.negative_prompt = before[neg.end():].strip()
    else:
        md.positive_prompt = before.strip()

    if settings_text:
        md.settings = _parse_forge_settings(settings_text)
        size = md.settings.get('Size', '')
        sm = re.match(r'(\d+)\s*x\s*(\d+)', size)
        if sm:
            md.width = int(sm.group(1))
            md.height = int(sm.group(2))

    return md


# ---------------------------------------------------------------------------
# ComfyUI parser (best-effort over the volatile prompt graph)
# ---------------------------------------------------------------------------

_TEXT_ENCODE_HINTS = ('TextEncode', 'CLIPTextEncode', 'PromptEncoder')
_SAMPLER_HINTS = ('Sampler', 'KSampler')
_CHECKPOINT_HINTS = ('CheckpointLoader', 'UNETLoader', 'ModelLoader')
_LATENT_SIZE_HINTS = ('EmptyLatentImage', 'EmptySD3LatentImage', 'EmptyImage')
_SAMPLER_KEYS = ('steps', 'cfg', 'seed', 'sampler_name', 'scheduler',
                 'denoise', 'eta', 'sampler_mode')


def _is_link(value) -> TypeGuard[list]:
    return isinstance(value, list) and len(value) >= 1 and not isinstance(
        value[0], (dict, list))


def _walk_back(prompt: dict, start_ids: set[str],
               match_class: tuple[str, ...]) -> list[dict]:
    """BFS through input links, returning nodes whose class_type matches."""
    found, seen, stack = [], set(), list(start_ids)
    while stack:
        nid = str(stack.pop())
        if nid in seen:
            continue
        seen.add(nid)
        node = prompt.get(nid)
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type', '')
        if any(h in ct for h in match_class):
            found.append(node)
            continue  # don't walk past a matching node
        inputs = node.get('inputs', {})
        if isinstance(inputs, dict):
            for v in inputs.values():
                if _is_link(v):
                    stack.append(str(v[0]))
    return found


def _extract_text_from_encoder(node: dict) -> str:
    inputs = node.get('inputs', {})
    if not isinstance(inputs, dict):
        return ''
    for key in ('text', 'prompt', 'string', 'positive', 'negative'):
        v = inputs.get(key)
        if isinstance(v, str):
            return v
    for v in inputs.values():
        if isinstance(v, str) and len(v) > 0:
            return v
    return ''


def parse_comfyui(prompt_text: str, workflow_text: str = '') -> GenMetadata:
    md = GenMetadata(format='comfyui', raw_text=prompt_text)
    try:
        prompt = json.loads(prompt_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return md
    if not isinstance(prompt, dict):
        return md

    if workflow_text:
        try:
            md.workflow = json.loads(workflow_text)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    pos_seed_ids: set[str] = set()
    neg_seed_ids: set[str] = set()

    # Find sampler nodes; trace their positive/negative inputs back to encoders.
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type', '')
        inputs = node.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        if any(h in ct for h in _SAMPLER_HINTS):
            pos = inputs.get('positive')
            neg = inputs.get('negative')
            if _is_link(pos):
                pos_seed_ids.add(str(pos[0]))
            if _is_link(neg):
                neg_seed_ids.add(str(neg[0]))
            for k in _SAMPLER_KEYS:
                v = inputs.get(k)
                if v is not None and not isinstance(v, list):
                    md.settings.setdefault(k, v)

    # Some workflows have no recognizable sampler; fall back to title-based
    # heuristic on CLIPTextEncode nodes.
    if not pos_seed_ids and not neg_seed_ids:
        for nid, node in prompt.items():
            if not isinstance(node, dict):
                continue
            ct = node.get('class_type', '')
            if not any(h in ct for h in _TEXT_ENCODE_HINTS):
                continue
            title = ''
            meta = node.get('_meta')
            if isinstance(meta, dict):
                title = str(meta.get('title', ''))
            tl = title.lower()
            if 'negative' in tl:
                neg_seed_ids.add(str(nid))
            elif 'positive' in tl or 'prompt' in tl:
                pos_seed_ids.add(str(nid))

    pos_nodes = _walk_back(prompt, pos_seed_ids, _TEXT_ENCODE_HINTS)
    neg_nodes = _walk_back(prompt, neg_seed_ids, _TEXT_ENCODE_HINTS)
    md.positive_prompt = '\n'.join(
        s for s in (_extract_text_from_encoder(n) for n in pos_nodes) if s
    ).strip()
    md.negative_prompt = '\n'.join(
        s for s in (_extract_text_from_encoder(n) for n in neg_nodes) if s
    ).strip()

    # Models / LoRAs / VAE / size — scan all nodes.
    models, loras, vaes, sizes = [], [], [], []
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type', '')
        inputs = node.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        if any(h in ct for h in _CHECKPOINT_HINTS):
            for k in ('ckpt_name', 'unet_name', 'model_name', 'model_path'):
                v = inputs.get(k)
                if isinstance(v, str):
                    models.append(v)
        if 'Lora' in ct or 'LoRA' in ct:
            name = inputs.get('lora_name') or inputs.get('name')
            if isinstance(name, str):
                strength = inputs.get('strength_model',
                                      inputs.get('strength'))
                if strength is None:
                    loras.append(name)
                else:
                    loras.append(f'{name} ({strength})')
        if 'VAELoader' in ct:
            v = inputs.get('vae_name')
            if isinstance(v, str):
                vaes.append(v)
        if any(h in ct for h in _LATENT_SIZE_HINTS):
            w, h = inputs.get('width'), inputs.get('height')
            if isinstance(w, int) and isinstance(h, int):
                sizes.append((w, h))

    if models:
        md.settings['model'] = ', '.join(models)
    if loras:
        md.settings['loras'] = ', '.join(loras)
    if vaes:
        md.settings['vae'] = ', '.join(vaes)
    if sizes:
        md.width, md.height = sizes[0]
        md.settings['size'] = f'{md.width}x{md.height}'

    return md


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def parse_metadata(path: Path) -> GenMetadata:
    """Detect and parse generation metadata from a PNG file."""
    info = read_png_text(path)

    text = info.get('parameters')
    if isinstance(text, str) and text.strip():
        return parse_forge(text)

    prompt_text = info.get('prompt')
    if isinstance(prompt_text, str) and prompt_text.strip():
        workflow_text = info.get('workflow', '')
        return parse_comfyui(
            prompt_text,
            workflow_text if isinstance(workflow_text, str) else '',
        )

    return GenMetadata()
