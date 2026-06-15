from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return project_root().parents[0]


def legacy_src_root() -> Path:
    return repo_root() / "src"


def resolve_from_config(config_path: str | Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(config_path).expanduser().resolve().parent / candidate).resolve()
