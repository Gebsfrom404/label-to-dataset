"""Worker that stages selected files under flat names and zips them.

The Caption tab's "Zip Selected" builds a flat dataset archive (images +
caption .txt, no folders). Because flattening can collide basenames and tar
cannot rename members on the fly, files are first staged into a temp dir
under their resolved flat names (hardlink when possible, copy otherwise),
then archived with the system tar via a ``-T`` list file.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ltd.workers.base_worker import BaseWorker


class ZipWorker(BaseWorker):
    """Stage (src, flat_name) pairs into a temp dir and zip them."""

    def __init__(self, mapping, dest_path, tar_exe, staging_dir, parent=None):
        super().__init__(parent)
        # mapping: list[tuple[Path, str]] of (source file, flat archive name)
        self._mapping = mapping
        self._dest_path = Path(dest_path)
        self._tar_exe = tar_exe
        self._staging_dir = Path(staging_dir)

    def do_work(self):
        total = len(self._mapping)
        self.status.emit(f'Staging {total} file(s)...')

        list_name = None
        try:
            # Stage each file under its flat name. Hardlinks are instant and
            # cost no extra disk; fall back to a copy across volumes or on
            # filesystems without hardlink support.
            for i, (src, flat_name) in enumerate(self._mapping):
                if self.is_cancelled:
                    return
                target = self._staging_dir / flat_name
                try:
                    os.link(src, target)
                except OSError:
                    shutil.copy2(src, target)
                self.progress.emit(i + 1, total)

            self.status.emit('Creating archive...')

            # bsdtar reads the -T list in the OS code page on Windows, so write
            # it in that encoding for non-ASCII (e.g. Cyrillic) names to work.
            list_encoding = 'mbcs' if sys.platform == 'win32' else 'utf-8'
            names = [flat_name for _, flat_name in self._mapping]
            list_path = self._staging_dir / '_ltd_zip_list.txt'
            list_name = str(list_path)
            with open(list_name, 'w', encoding=list_encoding,
                      errors='replace') as fh:
                fh.write('\n'.join(names))

            # -a picks the format from the .zip extension; -C sets the staging
            # dir as the base and -T reads the (flat) member names from it.
            cmd = [self._tar_exe, '-a', '-c', '-f', str(self._dest_path),
                   '-C', str(self._staging_dir), '-T', list_name]

            kwargs = {}
            creation_flag = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            if creation_flag:
                kwargs['creationflags'] = creation_flag

            result = subprocess.run(
                cmd, capture_output=True, text=True, **kwargs)
            if result.returncode != 0:
                msg = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f'tar failed (exit {result.returncode}):\n{msg}')
        finally:
            # Drop the staging copies/links; they can be large.
            shutil.rmtree(self._staging_dir, ignore_errors=True)
