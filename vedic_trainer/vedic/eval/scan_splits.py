"""The split names, in one place.

`vedic/eval/__init__.py` deliberately avoids importing `scan.py`/`cogs.py` at
package-import time, because those pull in transformers and datasets. That is
why the names were duplicated into the package init -- and why the two copies
drifted. They live here instead: no heavy imports, one definition, imported
by both the package init and the evaluators that validate against them.
"""
from __future__ import annotations

SCAN_SPLITS: tuple[str, ...] = ("simple", "length", "addprim_jump")
COGS_SPLITS: tuple[str, ...] = ("test", "gen")
