"""On-disk cache for prepared local-vol tables.

One JSON file per table under ``<output root>/local_vol/<index>/``, named by a
hash of the **inputs**, holding the coefficients next to the inputs they were
built from (which fit run / surface, which curve runs, which time grid, which
discretisation) and the time they were written - so a quote can be traced to
exactly what produced it, and a stale file can be checked (or deleted) by opening
it.  A table is **per index** (it is Dupire of that index's surface, discounted off
that index's borrow curve), so the index is a folder *and* a field of the
fingerprint - a file that travelled between two indices is caught by both.

The **spot** the table was built around is one of those inputs, in two places: the
fingerprint (``inputs.spot_anchor`` - it names the file) and the table itself
(``table.spot_anchor``).  That is not decoration: the coefficients are quoted in
log-moneyness *relative to that spot*, so a table built at another one is another
model.  :meth:`LocalVolFileCache.read` therefore refuses a file whose table does
not agree with the key it was found under - the caller rebuilds and the file is
rewritten - which also covers a hash collision and a hand-edited file.

``price-json`` wires it in through ``QUICK_DEFAULTS["local_vol_cache"]`` /
``--no-local-vol-cache``: a hit loads the coefficients, a miss builds and stores.
The same store object is used by every market of one run, which is what makes a
greeks run - and a whole spot ladder - pay for **one** table
(see :class:`surface_pricer.pricing.models.localvol.LocalVolCache`).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .fit_runs import bare_code, default_output_root

#: Folder name under the output root (next to ``vol_fit`` / ``ir_curve`` ...).
DIRECTORY_NAME = "local_vol"


def local_vol_root(
    output_root: Union[str, Path, None] = None, *, index: Any = None
) -> Path:
    """``<output root>/local_vol`` - and ``.../<bare index>`` when one is given.

    A local-vol table belongs to **one index** (it is Dupire of that index's fitted
    surface, discounted off that index's borrow curve), so the files are filed per
    index, exactly like the three run folders: browsing, comparing or discarding a
    table is a question about one index.
    """
    root = Path(output_root) if output_root else default_output_root()
    root = root / DIRECTORY_NAME
    key = bare_code(index) if index is not None else ""
    return root / key if key else root


def table_digest(fingerprint: Mapping[str, Any]) -> str:
    """The stable short hash of an input description (the file name's stem)."""
    text = json.dumps(dict(fingerprint), sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _same_spot(left: Any, right: Any) -> bool:
    """Two spots are the same spot, up to the rounding of a JSON round trip."""
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return abs(left_value - right_value) <= 1e-9 * max(1.0, abs(right_value))


class LocalVolFileCache:
    """The store the engines' ``LocalVolCache`` reads and writes through.

    Duck-typed on purpose: ``read(fingerprint) -> payload|None`` and
    ``write(fingerprint, payload)``, so the pricing layer never imports this
    module (layering: ``io`` depends on ``pricing``, not the other way round).
    """

    def __init__(
        self,
        output_root: Union[str, Path, None] = None,
        *,
        index: Any = None,
        enabled: bool = True,
    ):
        #: The index this store's tables belong to (``""`` = unfiled, a library
        #: caller that does not name one - the apps always do).
        self.index = bare_code(index) if index is not None else ""
        self.root = local_vol_root(output_root, index=self.index or None)
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.last_path: Optional[Path] = None
        self.last_created_at = ""

    # ------------------------------------------- the store protocol
    def read(self, fingerprint: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """The stored table for ``fingerprint``, or ``None`` (miss / cache off).

        A miss is not only "no file": the file has to **agree with what is being
        asked** - the inputs must be the ones in the fingerprint, and the table
        must have been built around the same spot.  A caller treats both the same
        way (build, and rewrite the file), so a mismatch heals itself.
        """
        if not self.enabled:
            return None
        path = self.path_for(fingerprint)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        # The inputs travel with the coefficients: a hash collision - or a file
        # somebody edited - must not be served as a table built for something else.
        if payload.get("inputs") != dict(fingerprint):
            self.misses += 1
            return None
        table = payload.get("table")
        if not isinstance(table, Mapping):
            self.misses += 1
            return None
        # ... and the table has to say which **spot** it was built around: the
        # coefficients are log-moneyness relative to it, so a table that names
        # another spot - or none at all - is not this table.
        wanted = fingerprint.get("spot_anchor")
        stored = table.get("spot_anchor")
        if wanted is not None and (stored is None or not _same_spot(stored, wanted)):
            self.misses += 1
            return None
        self.hits += 1
        self.last_path = path
        self.last_created_at = str(payload.get("created_at", ""))
        return dict(table)

    def write(
        self, fingerprint: Mapping[str, Any], table: Mapping[str, Any]
    ) -> Optional[Path]:
        """Store ``table`` under ``fingerprint`` (no-op when the cache is off)."""
        if not self.enabled:
            return None
        path = self.path_for(fingerprint)
        created_at = datetime.now().isoformat(sep=" ", timespec="seconds")
        payload = {
            "kind": "local_vol_table",
            "version": 1,
            "created_at": created_at,
            "inputs": dict(fingerprint),
            "table": dict(table),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        self.writes += 1
        self.last_path = path
        self.last_created_at = created_at
        return path

    def path_for(self, fingerprint: Mapping[str, Any]) -> Path:
        return self.root / "lv_{}.json".format(table_digest(fingerprint))

    # ------------------------------------------- reporting
    def describe(self, cache: Any = None) -> str:
        """One stderr line: what the table cost and where it came from.

        ``cache`` is the engines' ``LocalVolCache`` (it counts builds and loads);
        without it the line only reports the store side.
        """
        if not self.enabled:
            return "local vol : cache off (--no-local-vol-cache), table rebuilt"
        parts = []
        if cache is not None:
            parts.append("{} built".format(int(getattr(cache, "builds", 0) or 0)))
            parts.append("{} from cache".format(int(getattr(cache, "loads", 0) or 0)))
            stale = int(getattr(cache, "stale", 0) or 0)
            if stale:
                # a file that was not built around this spot: rebuilt (and rewritten)
                parts.append("{} stale (another spot, rebuilt)".format(stale))
        if self.last_path is not None:
            parts.append(
                "{} (written {})".format(self.last_path.name, self.last_created_at)
            )
        return "local vol : " + ", ".join(parts) if parts else "local vol : no table"


__all__ = [
    "DIRECTORY_NAME",
    "LocalVolFileCache",
    "local_vol_root",
    "table_digest",
]
