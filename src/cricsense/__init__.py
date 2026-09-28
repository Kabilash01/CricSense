"""CricSense: cricket video analysis."""

import os

# ultralytics otherwise pip-installs missing packages in the middle of a run. Dependencies
# are declared in pyproject.toml instead; a missing one should fail loudly, not self-install.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

__version__ = "0.1.0"
