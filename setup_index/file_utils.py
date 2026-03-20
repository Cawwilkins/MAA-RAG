from pathlib import Path

def get_source_id(file_path: str | Path, root_path: str | Path) -> str:
    return Path(file_path).resolve().relative_to(Path(root_path).resolve()).as_posix()