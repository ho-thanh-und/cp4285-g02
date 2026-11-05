"""Small file operations for resumable, auditable experiment stages."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .samples import file_hash


def write_json(path, value):
    """Replace a JSON file only after its full content has been written."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_table(frame, path):
    """Write CSV or Parquet through a temporary file so an interrupted run can retry."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_files(root, fingerprints):
    """Stop if an input or completed stage output has changed since it was recorded."""
    root = Path(root)
    for name, expected in fingerprints.items():
        if not (root / name).is_file() or file_hash(root / name) != expected:
            raise ValueError(f"File changed or missing: {root / name}")


def stage_complete(folder, stage):
    """A stage can be reused only when all recorded output files still match."""
    path = folder / f"{stage}_complete.json"
    if not path.exists():
        return False
    verify_files(folder, json.loads(path.read_text())["files"])
    return True


def finish_stage(folder, stage, filenames):
    """Mark completion last. Partial stage outputs can be safely rebuilt on a retry."""
    write_json(folder / f"{stage}_complete.json", dict(
        files={name: file_hash(folder / name) for name in filenames},
    ))


@contextmanager
def study_lock(folder):
    """Prevent concurrent writers. A lock left after a process crash is reported clearly."""
    lock = folder / "running.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Study is already running. If its process stopped, remove {lock} before resuming.") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)
