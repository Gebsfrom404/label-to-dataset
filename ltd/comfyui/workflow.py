"""ComfyUI workflow validation and manipulation."""
import json
from pathlib import Path

import requests

# Expected node title prefixes for LTD integration
LTD_INPUT_IMAGE = 'LTD_Input_Image'
LTD_INPUT_MASK = 'LTD_Input_Mask'
LTD_OUTPUT_IMAGE = 'LTD_Output_Image'
LTD_OUTPUT_MASK = 'LTD_Output_Mask'
LTD_OUTPUT_TEXT = 'LTD_Output_Text'

# Values for ComfyUI's internal seed control widget (not an API input)
_SEED_CONTROL_VALUES = {'fixed', 'increment', 'decrement', 'randomize'}


def load_workflow(source: str) -> dict | None:
    """Load workflow from JSON string or file path.

    Auto-converts UI format to API format if ComfyUI is running.
    """
    workflow = None
    try:
        path = Path(source)
        if path.exists() and path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
    except (OSError, ValueError):
        pass
    if workflow is None:
        try:
            workflow = json.loads(source)
        except (json.JSONDecodeError, TypeError):
            return None

    if workflow and not is_api_format(workflow) and _is_ui_format(workflow):
        converted = _convert_ui_to_api(workflow)
        if converted and is_api_format(converted):
            return converted

    return workflow


def is_api_format(workflow: dict) -> bool:
    """Check if workflow is in ComfyUI API format (numbered nodes)."""
    if not workflow:
        return False
    for key, value in workflow.items():
        if not isinstance(value, dict) or 'class_type' not in value:
            return False
    return True


def _is_ui_format(workflow: dict) -> bool:
    """Check if workflow is in ComfyUI UI format (nodes + links arrays)."""
    return (isinstance(workflow.get('nodes'), list) and
            isinstance(workflow.get('links'), list))


def _convert_ui_to_api(workflow: dict) -> dict | None:
    """Convert UI-format workflow to API format using ComfyUI's object_info.

    Returns None if ComfyUI is not reachable or conversion fails.
    """
    from ltd.settings import get_settings, DEFAULT_SETTINGS

    settings = get_settings()
    base_url = settings.value(
        'comfyui_url', DEFAULT_SETTINGS['comfyui_url'], type=str).rstrip('/')

    try:
        r = requests.get(f'{base_url}/object_info', timeout=10)
        if r.status_code != 200:
            return None
        object_info = r.json()
    except Exception:
        return None

    nodes = workflow.get('nodes', [])
    links = workflow.get('links', [])

    # Build link map: link_id -> (from_node_id_str, from_output_slot)
    link_map = {}
    for link in links:
        link_id = link[0]
        link_map[link_id] = (str(link[1]), link[2])

    api = {}
    for node in nodes:
        node_id = str(node['id'])
        class_type = node.get('type', '')
        title = node.get('title') or class_type

        # Skip utility-only nodes
        if class_type in ('Reroute', 'Note', 'PrimitiveNode'):
            continue

        info = object_info.get(class_type)
        if info is None:
            continue

        input_defs = info.get('input', {})
        required = input_defs.get('required', {})
        optional = input_defs.get('optional', {})

        api_inputs = {}

        # 1. Map linked (slot) inputs
        linked_names = set()
        for inp in node.get('inputs', []):
            name = inp['name']
            link_id = inp.get('link')
            if link_id is not None and link_id in link_map:
                from_id, from_slot = link_map[link_id]
                api_inputs[name] = [from_id, from_slot]
            linked_names.add(name)

        # 2. Map widget values to non-linked params using object_info order
        widgets_values = node.get('widgets_values') or []
        w_idx = 0

        all_params = list(required.items()) + list(optional.items())
        for param_name, param_def in all_params:
            if param_name in linked_names:
                continue

            if (not isinstance(param_def, (list, tuple))
                    or len(param_def) == 0):
                continue

            type_info = param_def[0]

            # Determine if this param has a widget (vs being slot-only)
            is_widget = False
            if isinstance(type_info, list):
                # Combo/dropdown: [["option1", "option2", ...], {...}]
                is_widget = True
            elif isinstance(type_info, str):
                if type_info in ('INT', 'FLOAT', 'STRING', 'BOOLEAN'):
                    is_widget = True
                # Else it's a slot type like IMAGE, MODEL, CLIP, etc.

            if is_widget and w_idx < len(widgets_values):
                api_inputs[param_name] = widgets_values[w_idx]
                w_idx += 1

                # Skip control_after_generate (internal widget after seed)
                if (param_name in ('seed', 'noise_seed')
                        and w_idx < len(widgets_values)):
                    next_val = widgets_values[w_idx]
                    if (isinstance(next_val, str)
                            and next_val in _SEED_CONTROL_VALUES):
                        w_idx += 1

        api[node_id] = {
            'class_type': class_type,
            'inputs': api_inputs,
            '_meta': {'title': title},
        }

    return api if api else None


def find_nodes_by_title(workflow: dict, title_prefix: str) -> list[str]:
    """Find node IDs whose _meta.title starts with the given prefix."""
    found = []
    for node_id, node_data in workflow.items():
        meta = node_data.get('_meta', {})
        title = meta.get('title', '')
        if title.startswith(title_prefix):
            found.append(node_id)
    return found


def validate_detection_workflow(workflow: dict) -> tuple[bool, str]:
    """Validate a workflow has required nodes for detection.

    Requires: LTD_Input_Image, and at least one of LTD_Output_Mask or LTD_Output_Image.
    """
    if not is_api_format(workflow):
        if _is_ui_format(workflow):
            return False, ('Workflow is in UI format and auto-conversion '
                           'failed. In ComfyUI: Settings \u2192 Enable Dev '
                           'Mode Options \u2192 use "Save (API Format)" button')
        return False, 'Workflow is not in valid ComfyUI format'

    input_nodes = find_nodes_by_title(workflow, LTD_INPUT_IMAGE)
    if not input_nodes:
        return False, f'Missing node with title "{LTD_INPUT_IMAGE}"'

    output_mask = find_nodes_by_title(workflow, LTD_OUTPUT_MASK)
    output_image = find_nodes_by_title(workflow, LTD_OUTPUT_IMAGE)
    if not output_mask and not output_image:
        return False, (f'Missing output node: need "{LTD_OUTPUT_MASK}" '
                       f'or "{LTD_OUTPUT_IMAGE}"')

    return True, 'OK'


def validate_modification_workflow(workflow: dict) -> tuple[bool, str]:
    """Validate a workflow for image modification.

    Requires: LTD_Input_Image, LTD_Input_Mask, LTD_Output_Image.
    """
    if not is_api_format(workflow):
        if _is_ui_format(workflow):
            return False, ('Workflow is in UI format and auto-conversion '
                           'failed. In ComfyUI: Settings \u2192 Enable Dev '
                           'Mode Options \u2192 use "Save (API Format)" button')
        return False, 'Workflow is not in valid ComfyUI format'

    if not find_nodes_by_title(workflow, LTD_INPUT_IMAGE):
        return False, f'Missing node with title "{LTD_INPUT_IMAGE}"'
    if not find_nodes_by_title(workflow, LTD_INPUT_MASK):
        return False, f'Missing node with title "{LTD_INPUT_MASK}"'
    if not find_nodes_by_title(workflow, LTD_OUTPUT_IMAGE):
        return False, f'Missing node with title "{LTD_OUTPUT_IMAGE}"'

    return True, 'OK'


def validate_caption_workflow(workflow: dict) -> tuple[bool, str]:
    """Validate a workflow for captioning.

    Requires: LTD_Input_Image, LTD_Output_Text.
    """
    if not is_api_format(workflow):
        if _is_ui_format(workflow):
            return False, ('Workflow is in UI format and auto-conversion '
                           'failed. In ComfyUI: Settings \u2192 Enable Dev '
                           'Mode Options \u2192 use "Save (API Format)" button')
        return False, 'Workflow is not in valid ComfyUI format'

    if not find_nodes_by_title(workflow, LTD_INPUT_IMAGE):
        return False, f'Missing node with title "{LTD_INPUT_IMAGE}"'
    if not find_nodes_by_title(workflow, LTD_OUTPUT_TEXT):
        return False, f'Missing node with title "{LTD_OUTPUT_TEXT}"'

    return True, 'OK'


def set_input_image(workflow: dict, image_filename: str):
    """Set the input image filename in the workflow."""
    for node_id in find_nodes_by_title(workflow, LTD_INPUT_IMAGE):
        workflow[node_id]['inputs']['image'] = image_filename


def set_input_mask(workflow: dict, mask_filename: str):
    """Set the input mask filename in the workflow."""
    for node_id in find_nodes_by_title(workflow, LTD_INPUT_MASK):
        workflow[node_id]['inputs']['image'] = mask_filename
