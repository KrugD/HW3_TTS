import json
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(path):
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    """Write JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
