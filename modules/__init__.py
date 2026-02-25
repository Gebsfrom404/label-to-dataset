"""Plugin module discovery system."""
import importlib
import inspect
import os
import pkgutil
from pathlib import Path


def discover_modules(base_class, package_path: str) -> list:
    """Discover all modules implementing base_class in the given package path.

    Args:
        base_class: The ABC to search for subclasses of
        package_path: Dot-separated or slash-separated package path
                      e.g. 'modules/detection' or 'modules.detection'
    """
    modules = []
    # Normalize to dot-notation
    package_name = package_path.replace('/', '.').replace('\\', '.')

    # Try to find the directory
    # First, resolve relative to the working directory
    dir_path = Path(package_path.replace('.', os.sep))
    if not dir_path.is_absolute():
        # Try from cwd
        candidates = [
            Path.cwd() / dir_path,
            Path(__file__).parent.parent / dir_path,
        ]
        dir_path = None
        for c in candidates:
            if c.exists():
                dir_path = c
                break

    if dir_path is None or not dir_path.exists():
        return modules

    for finder, name, is_pkg in pkgutil.iter_modules([str(dir_path)]):
        if name.startswith('_'):
            continue
        try:
            module = importlib.import_module(f'{package_name}.{name}')
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, base_class) and obj is not base_class
                        and not inspect.isabstract(obj)):
                    modules.append(obj())
        except Exception:
            import traceback
            traceback.print_exc()
            continue
    return modules
