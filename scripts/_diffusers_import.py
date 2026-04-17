import os
import sys
from pathlib import Path
from typing import Optional


def setup_diffusers_import_path(diffusers_src: Optional[str] = None) -> Optional[str]:
    """
    Optionally prepend a local diffusers source tree to sys.path.
    Priority:
      1) explicit --diffusers_src
      2) DIFFUSERS_SRC env
      3) <repo>/external/diffusers/src
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidate = diffusers_src or os.environ.get("DIFFUSERS_SRC")
    if candidate is None:
        candidate = str(repo_root / "external" / "diffusers" / "src")

    candidate_path = Path(candidate).resolve()
    if candidate_path.exists() and candidate_path.is_dir():
        candidate_str = str(candidate_path)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return candidate_str
    return None
