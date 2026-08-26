"""Minimal stand-in for the handful of `comfy.*` symbols the vendored SAM3
files (`modules/sam3/vendor/`) import.

The reference implementation is a ComfyUI custom node; its model code only
touches ComfyUI for a few thin torch wrappers (mixed-precision layer
construction, one dtype-cast helper, device selection, checkpoint loading,
and attention dispatch) — never nodes, graphs, or the ModelPatcher. This
module supplies plain-PyTorch equivalents for exactly those symbols and
registers them into ``sys.modules`` so the vendored files' ``import
comfy.ops`` etc. resolve without ComfyUI installed.

Must be imported (via :func:`install`) before any `modules.sam3.vendor.*`
module is imported.
"""
import sys
import types

import torch
import torch.nn.functional as F


def _attention_pytorch(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False):
    """Plain scaled_dot_product_attention, matching ComfyUI's calling
    convention (q/k/v as [B, H, L, D] when skip_reshape=True, else
    [B, L, H*D])."""
    if not skip_reshape:
        b, s, _ = q.shape
        q = q.view(b, s, heads, -1).transpose(1, 2)
        b, s, _ = k.shape
        k = k.view(b, s, heads, -1).transpose(1, 2)
        b, s, _ = v.shape
        v = v.view(b, s, heads, -1).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if not skip_output_reshape:
        b, h, s, d = out.shape
        out = out.transpose(1, 2).reshape(b, s, h * d)
    return out


def _optimized_attention_for_device(device, mask=False):
    return _attention_pytorch


def _cast_to_input(tensor, like):
    return tensor.to(like.dtype)


def _get_torch_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_torch_file(path, safe_load=True, device=None):
    path = str(path)
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path, device='cpu' if device is None else str(device))
    return torch.load(path, map_location='cpu', weights_only=True)


class _ProgressBar:
    def __init__(self, total):
        self.total = total

    def update_absolute(self, value, total=None, preview=None):
        pass

    def update(self, value):
        pass


def install():
    """Register fake `comfy.*` modules into sys.modules (idempotent)."""
    if 'comfy' in sys.modules:
        return

    comfy = types.ModuleType('comfy')

    ops_mod = types.ModuleType('comfy.ops')
    manual_cast = types.ModuleType('comfy.ops.manual_cast')
    manual_cast.Linear = torch.nn.Linear
    manual_cast.Conv1d = torch.nn.Conv1d
    manual_cast.Conv2d = torch.nn.Conv2d
    manual_cast.Conv3d = torch.nn.Conv3d
    manual_cast.ConvTranspose2d = torch.nn.ConvTranspose2d
    manual_cast.LayerNorm = torch.nn.LayerNorm
    manual_cast.Embedding = torch.nn.Embedding
    manual_cast.GroupNorm = torch.nn.GroupNorm
    ops_mod.manual_cast = manual_cast
    ops_mod.cast_to_input = _cast_to_input

    model_management_mod = types.ModuleType('comfy.model_management')
    model_management_mod.get_torch_device = _get_torch_device

    utils_mod = types.ModuleType('comfy.utils')
    utils_mod.load_torch_file = _load_torch_file
    utils_mod.ProgressBar = _ProgressBar

    ldm_mod = types.ModuleType('comfy.ldm')
    ldm_modules_mod = types.ModuleType('comfy.ldm.modules')
    ldm_attention_mod = types.ModuleType('comfy.ldm.modules.attention')
    ldm_attention_mod.optimized_attention_for_device = _optimized_attention_for_device
    ldm_attention_mod.attention_pytorch = _attention_pytorch

    comfy.ops = ops_mod
    comfy.model_management = model_management_mod
    comfy.utils = utils_mod
    comfy.ldm = ldm_mod
    ldm_mod.modules = ldm_modules_mod
    ldm_modules_mod.attention = ldm_attention_mod

    sys.modules['comfy'] = comfy
    sys.modules['comfy.ops'] = ops_mod
    sys.modules['comfy.ops.manual_cast'] = manual_cast
    sys.modules['comfy.model_management'] = model_management_mod
    sys.modules['comfy.utils'] = utils_mod
    sys.modules['comfy.ldm'] = ldm_mod
    sys.modules['comfy.ldm.modules'] = ldm_modules_mod
    sys.modules['comfy.ldm.modules.attention'] = ldm_attention_mod
