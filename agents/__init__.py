"""Agent package with unified external dependency management."""

import sys
from pathlib import Path

# Add external/ directory to sys.path for all external dependencies
EXTERNAL_PATH = Path(__file__).parents[1] / "external"

if EXTERNAL_PATH.exists():
    # Add paths for external dependencies
    external_paths = [
        EXTERNAL_PATH,  # For 'from gam import ...'
        EXTERNAL_PATH / "memalpha",  # For 'from agent import ...', 'from memory import ...'
        EXTERNAL_PATH / "mem1",  # For 'from inference.data_pipelines import ...'
        EXTERNAL_PATH / "verl",  # For 'from verl.experimental... import ...'
    ]

    for path in external_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

__all__: list[str] = []