"""Simplified ComfyUI HTTP API + websocket client."""
import json
import uuid
from pathlib import Path

import requests
import websocket

from ltd.settings import get_settings, DEFAULT_SETTINGS


class ComfyUIClient:
    """Client for communicating with a running ComfyUI instance."""

    def __init__(self, base_url: str | None = None):
        if base_url is None:
            settings = get_settings()
            base_url = settings.value('comfyui_url',
                                      DEFAULT_SETTINGS['comfyui_url'],
                                      type=str)
        self.base_url = base_url.rstrip('/')
        self.client_id = str(uuid.uuid4())

    def health_check(self) -> bool:
        """Check if ComfyUI is reachable."""
        try:
            r = requests.get(f'{self.base_url}/history/123', timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def upload_image(self, image_path: Path, subfolder: str = '',
                     overwrite: bool = True) -> dict:
        """Upload an image to ComfyUI's input folder."""
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/png')}
            data = {'subfolder': subfolder, 'overwrite': str(overwrite).lower()}
            r = requests.post(f'{self.base_url}/upload/image',
                              files=files, data=data, timeout=30)
            return r.json()

    def free(self, unload_models: bool = True,
             free_memory: bool = True) -> bool:
        """Ask ComfyUI to drop models/cache from VRAM (best-effort)."""
        try:
            r = requests.post(
                f'{self.base_url}/free',
                json={'unload_models': unload_models,
                      'free_memory': free_memory}, timeout=30)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow and return the prompt_id."""
        payload = {'prompt': workflow, 'client_id': self.client_id}
        r = requests.post(f'{self.base_url}/prompt',
                          json=payload, timeout=30)
        result = r.json()
        if 'prompt_id' not in result:
            raise RuntimeError(f'Failed to queue prompt: {result}')
        return result['prompt_id']

    def wait_for_completion(self, prompt_id: str) -> dict:
        """Wait for workflow completion via websocket, return history."""
        host = self.base_url.replace('http://', '').replace('https://', '')
        ws = websocket.WebSocket()
        ws.connect(f'ws://{host}/ws?clientId={self.client_id}')
        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if (data['node'] is None and
                                data['prompt_id'] == prompt_id):
                            break
        finally:
            ws.close()

        # Fetch history
        r = requests.get(f'{self.base_url}/history/{prompt_id}', timeout=30)
        return r.json().get(prompt_id, {})

    def get_output_files(self, history: dict) -> list[dict]:
        """Extract output image/gif refs from history.

        Returns dicts with 'filename', 'subfolder' and 'type' — the subfolder
        matters because a SaveImage 'filename_prefix' like 'folder/name' writes
        into a subfolder, and /view needs it to locate the file.
        """
        files = []
        outputs = history.get('outputs', {})
        for node_id, node_output in outputs.items():
            for key in ('images', 'gifs'):
                for item in node_output.get(key, []):
                    files.append({
                        'filename': item['filename'],
                        'subfolder': item.get('subfolder', ''),
                        'type': item.get('type', 'output'),
                    })
        return files

    def get_output_text(self, history: dict) -> list[str]:
        """Extract text output from history."""
        texts = []
        outputs = history.get('outputs', {})
        for node_id, node_output in outputs.items():
            if 'text' in node_output:
                texts.extend(node_output['text'])
        return texts

    def download_output(self, filename: str, dest_dir: Path,
                        subfolder: str = '', file_type: str = 'output') -> Path:
        """Download an output file from ComfyUI (raises on HTTP error)."""
        params = {'filename': filename, 'subfolder': subfolder,
                  'type': file_type}
        r = requests.get(f'{self.base_url}/view', params=params, timeout=60)
        r.raise_for_status()
        dest = dest_dir / filename
        dest.write_bytes(r.content)
        return dest

    def run_workflow(self, workflow: dict, output_dir: Path) -> dict:
        """Queue workflow, wait, and download all outputs.

        Returns dict with 'files' (list of Paths) and 'texts' (list of str).
        """
        prompt_id = self.queue_prompt(workflow)
        history = self.wait_for_completion(prompt_id)

        files = []
        for ref in self.get_output_files(history):
            path = self.download_output(
                ref['filename'], output_dir,
                subfolder=ref['subfolder'], file_type=ref['type'])
            files.append(path)

        texts = self.get_output_text(history)
        return {'files': files, 'texts': texts}
