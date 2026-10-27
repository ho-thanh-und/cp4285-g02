"""Download the original public competition archive; never execute source scripts."""
import hashlib
import json
import re
import zipfile
from pathlib import Path
import requests

FOLDER = "https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw"
FILES = {
    "dataset-train-diginetica.zip": ("0B7XZSACQf0KdenRmMk8yVUU5LWc", "0-dIOA5Y_6oJMTVp2mTlVcEw"),
    "diginetica_baseline.py": ("0B7XZSACQf0KdU3JweDMzZUxyemM", "0-e-6jws54QH5p58k1l5z73Q"),
}


def download(name, file_id, resource_key):
    target = Path("data/source") / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (zipfile.is_zipfile(target) if name.endswith('.zip') else target.read_bytes().startswith(b'#')):
        return target
    session = requests.Session()
    url = "https://drive.usercontent.google.com/download"
    params = dict(id=file_id, export="download", resourcekey=resource_key)
    response = session.get(url, params=params, stream=True, timeout=(30, 180))
    response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        fields = dict(re.findall(r'name="([^"]+)" value="([^"]+)"', response.text))
        if fields.get("confirm") != "t" or "uuid" not in fields:
            raise RuntimeError("Public download unavailable; received HTML rather than data")
        response = session.get(url, params=fields, stream=True, timeout=(30, 180))
        response.raise_for_status()
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("wb") as stream:
        for chunk in response.iter_content(1024 * 1024):
            stream.write(chunk)
    if name.endswith('.zip') and not zipfile.is_zipfile(temporary):
        raise RuntimeError("Downloaded archive is not a ZIP")
    if name.endswith('.py') and not temporary.read_bytes().startswith(b'#'):
        raise RuntimeError("Expected original baseline source; received another format")
    temporary.replace(target)
    return target


def main():
    manifest = {"folder_url": FOLDER, "files": []}
    for name, (file_id, key) in FILES.items():
        target = download(name, file_id, key)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        manifest["files"].append(dict(name=name, file_id=file_id, bytes=target.stat().st_size, sha256=digest))
        if name.endswith(".zip"):
            raw = Path("data/raw").resolve()
            raw.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target) as archive:
                for entry in archive.infolist():
                    destination = (raw / entry.filename).resolve()
                    if not destination.is_relative_to(raw):
                        raise ValueError("Unsafe archive member")
                archive.extractall(raw)
    Path("data/source/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Original data verified and extracted. Manifest: data/source/manifest.json")


if __name__ == "__main__":
    main()
