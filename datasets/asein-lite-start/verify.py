"""Verify archived Asein inputs without importing the simulation."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    names = {"initial.json", "config.json", "original-artwork.json", "requested-config.json"}
    if set(manifest["files"]) != names:
        raise SystemExit("Unexpected input file inventory")
    data = {}
    for name in sorted(names):
        raw = (root / name).read_bytes()
        expected = manifest["files"][name]
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise SystemExit(f"Checksum or size mismatch: {name}")
        data[name] = json.loads(raw)
    initial = data["initial.json"]
    original = data["original-artwork.json"]
    config = data["config.json"]
    if (initial["width"], initial["height"]) != (512, 256):
        raise SystemExit("Unexpected map dimensions")
    if len(initial["crust"]) != 512 * 256 or initial["crust"] != original["crust"]:
        raise SystemExit("Initial artwork differs")
    if any(original[key] != value for key, value in initial.items()):
        raise SystemExit("Normalized map content differs from original artwork")
    if config != data["requested-config.json"]:
        raise SystemExit("Saved and requested configurations differ")
    if config["seed"] != 37 or config["effective_subduction"]["force_n_per_m"] != 3e13:
        raise SystemExit("Unexpected starting configuration")
    print("Verified 4 original input files: Asein Lite, seed 37, 512 x 256, force 3e13 N/m.")


if __name__ == "__main__":
    main()
