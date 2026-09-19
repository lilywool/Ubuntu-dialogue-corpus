"""Clear code-cell outputs and execution counts from Jupyter notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def clear_notebook(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    for cell in document.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: clear_notebook_outputs.py NOTEBOOK [NOTEBOOK ...]")
    for argument in sys.argv[1:]:
        path = Path(argument)
        clear_notebook(path)
        print(f"cleared {path}")


if __name__ == "__main__":
    main()
