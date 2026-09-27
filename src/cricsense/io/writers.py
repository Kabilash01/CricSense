"""Validated reads and writes of pass artifacts, plus logging setup.

Every JSON a pass writes goes through write_doc and every JSON it reads goes through
read_doc, so a schema violation fails at the boundary instead of three passes later.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

DocT = TypeVar("DocT", bound=BaseModel)


def write_doc(doc: BaseModel, path: Path) -> Path:
    """Write JSON atomically: a crash mid-write never leaves a truncated artifact behind."""
    return _write_atomic(path, doc.model_dump_json(indent=1))


def read_doc(path: Path, model: type[DocT]) -> DocT:
    _require(path)
    return model.model_validate_json(path.read_bytes())


def write_yaml_doc(doc: BaseModel, path: Path) -> Path:
    """Same as write_doc, as YAML, for files people read and check in (calibration)."""
    text = yaml.safe_dump(doc.model_dump(mode="json"), sort_keys=False, default_flow_style=None, width=100)
    return _write_atomic(path, text)


def read_yaml_doc(path: Path, model: type[DocT]) -> DocT:
    _require(path)
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the pass that produces it first")


def _write_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Configure the 'cricsense' logger: console always, file if log_file is given."""
    logger = logging.getLogger("cricsense")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
