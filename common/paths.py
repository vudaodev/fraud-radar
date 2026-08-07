from pathlib import Path

def _find_root(start: Path, marker: str = "pyproject.toml") -> Path:
    for d in [start, *start.parents]:
        if (d / marker).exists():
            return d
    raise FileNotFoundError(f"No {marker} found above {start}")

ROOT = _find_root(Path(__file__).resolve())
