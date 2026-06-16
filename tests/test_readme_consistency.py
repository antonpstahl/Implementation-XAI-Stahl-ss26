"""
Consistency check: README tables must match the artefacts in results/.

Run:  pytest tests/test_readme_consistency.py
Fix:  python utils/update_readme_tables.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "utils"))

from update_readme_tables import (  # noqa: E402
    ROOT,
    _TABLES,
    extract_table,
)


def _readme_cache() -> dict[Path, str]:
    seen: dict[Path, str] = {}
    for path, _, _ in _TABLES:
        if path not in seen:
            seen[path] = path.read_text(encoding="utf-8")
    return seen


_READMES = _readme_cache()


@pytest.mark.parametrize("readme_path,name,gen_fn", _TABLES, ids=[t[1] for t in _TABLES])
def test_table_matches_results(readme_path: Path, name: str, gen_fn) -> None:
    """Generated table must equal the sentinel block currently in the README."""
    current = extract_table(_READMES[readme_path], name)
    expected = gen_fn()
    assert current == expected, (
        f"\nTable '{name}' in {readme_path.name} is out of sync with results/.\n"
        f"Run:  python utils/update_readme_tables.py\n\n"
        f"--- README (current) ---\n{current}\n\n"
        f"--- Generated (expected) ---\n{expected}\n"
    )
