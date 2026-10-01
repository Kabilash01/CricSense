"""Object storage for what a video produces: the overlay video, images, audio, JSON outputs.

`local` (a directory; the default) or `s3` (any S3-compatible store - MinIO, AWS - through
boto3, with credentials from the environment as boto3 reads them). Keys are
<video_id>/<file name>; put() returns the stored file's URI and checksum.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stored:
    uri: str
    sha256: str
    bytes: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, path: Path, key: str) -> Stored:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        return Stored(dest.resolve().as_uri(), _sha256(dest), dest.stat().st_size)

    def open_path(self, uri: str) -> Path:
        return Path(uri.removeprefix("file://"))


class S3Store:
    def __init__(self, bucket: str, endpoint_url: str | None = None, prefix: str = "") -> None:
        import boto3

        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)

    def put(self, path: Path, key: str) -> Stored:
        full = f"{self.prefix}/{key}" if self.prefix else key
        self.s3.upload_file(str(path), self.bucket, full)
        return Stored(f"s3://{self.bucket}/{full}", _sha256(path), path.stat().st_size)


def open_store(cfg: dict, root: Path):
    backend = cfg.get("backend", "local")
    if backend == "local":
        return LocalStore(root / cfg.get("root", "outputs/store"))
    if backend == "s3":
        return S3Store(cfg["bucket"], cfg.get("endpoint_url"), cfg.get("prefix", ""))
    raise ValueError(f"unknown storage backend {backend!r}")
