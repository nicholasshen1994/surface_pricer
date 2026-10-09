"""The listed-option contract map: what a contract *is*, without the database.

Wind's ``WINDDF.CHINAOPTIONDESCRIPTION`` knows every listed option's strike,
expiry, call/put and contract unit.  The quote gateway does not: an SSE/SZSE ETF
option snapshot comes back keyed by the exchange's **numeric contract id**
(``10012493``) and nothing else, which is why ``510500`` used to fit nothing at
all - see the design note.  The two are joined on the contract's **Wind code**
(``10012493.SH`` / ``MO2505-C-5500.CFE``): the snapshot record's
``resp_stk_code`` plus its exchange suffix.

That join is fetched **once** into one local file
(``data/option_contracts.json``, written by
``python -m surface_pricer fetch-contracts``), so the pricing path never talks to
Oracle:

* **one file, full chains** - each venue's whole history, so a contract that
  expires is still readable (and the file only grows when a new contract appears);
* keyed by Wind code, each entry carrying ``underlying`` / ``code`` (the exchange's
  human-readable option code) / ``call_put`` / ``strike`` / ``maturity`` /
  ``unit``;
* an ``updated_at`` for the file and for each venue, so "is this still current?"
  is answered by reading it.

:func:`spec_for_record` is the provider's entry point: it turns a snapshot record
into the same :class:`ParsedListedOption` the code parsers return (the two families
are now read through one mechanism), and returns ``None`` when the contract is not
in the map - the caller then falls back to parsing the code.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.daycount import to_date, to_datetime
from .listed_contracts import (
    EXCHANGE_CFFEX,
    EXCHANGE_SSE,
    EXCHANGE_SZSE,
    ParsedListedOption,
)
from .registry import UnderlyingSpec, get_underlying_spec

# --------------------------------------------------------------------------- file
#: ``surface_pricer/data`` - next to ``interest_rate.csv``.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FILE_NAME = "option_contracts.json"
FILE_KIND = "option_contracts"
FILE_VERSION = 1


def contract_file(path: Any = None) -> Path:
    """The contract-map file (``--file`` overrides the packaged ``data/`` one)."""
    return Path(path) if path else DATA_DIR / FILE_NAME


# ---------------------------------------------------------------------- database
#: The Wind Oracle listener this desk's boxes talk to (the dayBreaker UAT env), the
#: service name, and the read-only account.  Everything is overridable with
#: ``WIND_DB_*`` environment variables (``surface_pricer/.env`` is loaded by the
#: CLI, so the password belongs there - not in a command line).
DEFAULT_DB_HOST = "10.113.208.103"
DEFAULT_DB_PORT = 1521
DEFAULT_DB_NAME = "van"
DEFAULT_DB_USER = "eq_edsqis_read"
#: Where an Instant Client may sit (the server is older than python-oracledb's
#: thin mode can talk to, so a thick client is required).  ``ORACLE_CLIENT_LIB_DIR``
#: overrides, and the search below is only used when it is unset.
ORACLE_LIB_CANDIDATES = (
    r"C:\python\Scripts",
    r"C:\oracle\instantclient",
    r"C:\instantclient",
    r"C:\app\client",
)

#: Wind's ``S_INFO_CALLPUT`` for the two option kinds (its last six digits).
_CALL_PUT_CODES = {"001000": "call", "002000": "put"}

#: The whole chain of one venue: ``S_INFO_SCCODE`` is the venue's **aggregate
#: option code** (``510500OP.SH``, ``159915OP.SZ``, ``MO.CFE``).  One table only -
#: the read-only account can see ``CHINAOPTIONDESCRIPTION`` but not the
#: contract-property table, and nothing here needs the latter.
_CHAIN_SQL = """
SELECT S_INFO_WINDCODE, S_INFO_EXCODE, S_INFO_CALLPUT,
       S_INFO_STRIKEPRICE, S_INFO_MATURITYDATE, S_INFO_COUNIT
  FROM WINDDF.CHINAOPTIONDESCRIPTION
 WHERE S_INFO_SCCODE = :venue
"""


def wind_venue_code(spec: UnderlyingSpec) -> str:
    """The code Wind files a venue's **option chain** under (its ``S_INFO_SCCODE``).

    CFFEX index options live under the venue itself (``MO.CFE``); an ETF chain
    under the ETF's aggregate option code (``510500OP.SH`` / ``159915OP.SZ``),
    which is the ETF's own code plus the exchange suffix of its ``spot_ticker``.
    """
    if spec.kind == "etf":
        code, _, suffix = str(spec.spot_ticker or "").partition(".")
        return "{}OP.{}".format(code or spec.underlying, suffix or "SH")
    return "{}.CFE".format(spec.underlying)


def wind_db_url() -> str:
    """``user/password@host:port/service`` from ``WIND_DB_*`` (with sane defaults).

    ``WIND_DB_URL`` wins when it is set, so a box that already knows its Wind
    connection string needs no new variables.
    """
    url = os.environ.get("WIND_DB_URL", "").strip()
    if url:
        return url
    user = os.environ.get("WIND_DB_USER", "").strip() or DEFAULT_DB_USER
    password = os.environ.get("WIND_DB_PASSWORD", "")
    host = os.environ.get("WIND_DB_HOST", "").strip() or DEFAULT_DB_HOST
    port = os.environ.get("WIND_DB_PORT", "").strip() or str(DEFAULT_DB_PORT)
    name = os.environ.get("WIND_DB_NAME", "").strip() or DEFAULT_DB_NAME
    if not password:
        raise RuntimeError(
            "no Wind password: set WIND_DB_PASSWORD (or WIND_DB_URL) - the CLI "
            "reads surface_pricer/.env"
        )
    # the desk's spelling: user/password@host:port/service (see _split_url)
    return "{}/{}@{}:{}/{}".format(user, password, host, port, name)


def _split_url(url: str) -> Tuple[str, str, str, str, str]:
    """Split ``user/password@host:port/service`` - the password may contain anything."""
    credentials, _, address = str(url).rpartition("@")
    if not credentials or not address:
        raise ValueError(
            "Wind connection string must look like user/password@host:port/service"
        )
    user, _, password = credentials.partition("/")
    host_port, _, service = address.partition("/")
    host, _, port = host_port.partition(":")
    return user, password, host, port or str(DEFAULT_DB_PORT), service or DEFAULT_DB_NAME


def connect(url: Optional[str] = None):
    """A Wind connection - thin mode first, thick when the server is too old.

    python-oracledb's thin mode refuses this server (``DPY-3010``: older than
    12.1) and ``cx_Oracle`` needs a client library as well, so the fallback walks
    the usual Instant Client locations (or ``ORACLE_CLIENT_LIB_DIR``).
    """
    try:
        import oracledb
    except ImportError as error:  # pragma: no cover - depends on the box
        raise RuntimeError(
            "fetch-contracts needs python-oracledb (pip install oracledb)"
        ) from error

    user, password, host, port, service = _split_url(url or wind_db_url())
    dsn = oracledb.makedsn(host, int(port), service_name=service)
    try:
        return oracledb.connect(
            user=user, password=password, dsn=dsn, tcp_connect_timeout=15
        )
    except Exception as error:  # noqa: BLE001 - thin mode can fail for one reason
        if "DPY-3010" not in str(error):
            raise
    if not _init_thick_client(oracledb):
        raise RuntimeError(
            "the Wind server needs a thick Oracle client; set ORACLE_CLIENT_LIB_DIR "
            "to its directory, or install python-oracledb in thick mode"
        )
    return oracledb.connect(user=user, password=password, dsn=dsn)


def _init_thick_client(oracledb) -> bool:
    """Load an Instant Client (idempotent); ``False`` when none is found."""
    configured = os.environ.get("ORACLE_CLIENT_LIB_DIR", "").strip()
    candidates: Sequence[Optional[str]] = (
        (configured,) if configured else (None, *ORACLE_LIB_CANDIDATES)
    )
    for lib_dir in candidates:
        try:
            oracledb.init_oracle_client(**({"lib_dir": lib_dir} if lib_dir else {}))
            return True
        except Exception:  # noqa: BLE001 - try the next location
            if "already initialized" in str(_last_error()):  # pragma: no cover
                return True
            continue
    return False


def _last_error() -> str:  # pragma: no cover - only used for the init message
    import sys

    error = sys.exc_info()[1]
    return "" if error is None else str(error)


def _call_put(value: Any) -> str:
    """``708001000`` / ``708002000`` -> ``call`` / ``put`` (raise on anything else)."""
    text = str(int(value))
    resolved = _CALL_PUT_CODES.get(text[-6:])
    if resolved is None:
        raise ValueError(
            "unknown Wind option kind {!r}: expected the ...001000 (call) or "
            "...002000 (put) code".format(value)
        )
    return resolved


def _maturity(value: Any) -> date:
    """Wind's ``S_INFO_MATURITYDATE`` as a date (``20260422`` or a datetime)."""
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():  # the YYYYmmdd column value
        return datetime.strptime(text, "%Y%m%d").date()
    return to_date(text)


def fetch_chain(spec: UnderlyingSpec, *, url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read one venue's **whole** option chain (every contract Wind knows).

    One query, taken as given: rows are returned in the order the database gives
    them, ready for :meth:`ContractMap.merge`.
    """
    venue = wind_venue_code(spec)
    connection = connect(url)
    try:
        cursor = connection.cursor()
        cursor.execute(_CHAIN_SQL, [venue])
        rows = cursor.fetchall()
        cursor.close()
    finally:
        connection.close()
    entries: List[Dict[str, Any]] = []
    for windcode, excode, call_put, strike, maturity, unit in rows:
        entries.append(
            {
                "key": str(windcode).strip().upper(),
                "underlying": spec.underlying,
                "code": str(excode or "").strip().upper(),
                "call_put": _call_put(call_put),
                "strike": float(strike),
                "maturity": _maturity(maturity).isoformat(),
                "unit": None if unit is None else float(unit),
                "exchange": EXCHANGE_CFFEX if spec.kind != "etf" else spec.etf_exchange,
            }
        )
    return entries


# ------------------------------------------------------------------------- map
class ContractMap:
    """The local contract file: lookup + incremental merge."""

    def __init__(self, contracts=None, venues=None, updated_at=""):
        self.contracts: Dict[str, Dict[str, Any]] = dict(contracts or {})
        self.venues: Dict[str, Dict[str, Any]] = dict(venues or {})
        self.updated_at: str = str(updated_at or "")

    # ------------------------------------------------------------- lookup
    def lookup(self, key: str) -> Optional[Dict[str, Any]]:
        return self.contracts.get(str(key or "").strip().upper())

    def spec_for_key(self, key: str) -> Optional[ParsedListedOption]:
        """The parsed terms of one contract, or ``None`` when it is not in the map."""
        entry = self.lookup(key)
        if entry is None:
            return None
        return ParsedListedOption(
            raw_code=str(entry.get("code") or key),
            underlying=str(entry.get("underlying") or ""),
            exchange=str(entry.get("exchange") or ""),
            expiry=to_datetime(str(entry["maturity"])),
            option_type=str(entry["call_put"]),
            strike=float(entry["strike"]),
        )

    def unit(self, key: str) -> Optional[float]:
        """The contract unit (multipler) of one contract, when the file has it."""
        entry = self.lookup(key)
        return None if entry is None else entry.get("unit")

    # -------------------------------------------------------------- merge
    def merge(
        self, spec: UnderlyingSpec, entries: Iterable[Mapping[str, Any]], *, now=None
    ) -> Tuple[int, int]:
        """Add ``entries`` to the map; returns ``(added, total)`` for the venue.

        Existing keys are **left alone** (the file is the cache, the database the
        source of truth: a re-fetch that brings no new contract must not rewrite
        history), which is what makes "only new contracts update the file" true.
        """
        added = 0
        for item in entries:
            key = str(item["key"]).strip().upper()
            if key in self.contracts:
                continue
            self.contracts[key] = {
                "underlying": item["underlying"],
                "code": item["code"],
                "call_put": item["call_put"],
                "strike": float(item["strike"]),
                "maturity": item["maturity"],
                "unit": item.get("unit"),
                "exchange": item["exchange"],
            }
            added += 1
        stamp = (now or datetime.now()).isoformat(sep=" ", timespec="seconds")
        venue_code = wind_venue_code(spec)
        self.venues[venue_code] = {
            "underlying": spec.underlying,
            "chain_code": venue_code,
            "updated_at": stamp,
            "count": sum(
                1
                for entry in self.contracts.values()
                if str(entry.get("underlying")) == spec.underlying
            ),
        }
        if added:
            self.updated_at = stamp
        return added, int(self.venues[venue_code]["count"])

    # --------------------------------------------------------------- file
    def to_payload(self) -> Dict[str, Any]:
        return {
            "kind": FILE_KIND,
            "version": FILE_VERSION,
            "updated_at": self.updated_at,
            "venues": dict(sorted(self.venues.items())),
            "contracts": dict(sorted(self.contracts.items())),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ContractMap":
        if str(payload.get("kind") or FILE_KIND) != FILE_KIND:
            raise ValueError(
                "{} is not an {} file (kind={!r})".format(
                    FILE_NAME, FILE_KIND, payload.get("kind")
                )
            )
        return cls(
            contracts=payload.get("contracts") or {},
            venues=payload.get("venues") or {},
            updated_at=payload.get("updated_at") or "",
        )

    def save(self, path: Any = None) -> Path:
        target = contract_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_payload(), indent=1, sort_keys=True), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: Any = None) -> "ContractMap":
        """Read the file; a missing one is an **empty** map (nothing is guessed)."""
        target = contract_file(path)
        if not target.is_file():
            return cls()
        return cls.from_payload(json.loads(target.read_text(encoding="utf-8")))

    def describe(self) -> str:
        venues = ", ".join(
            "{}={}".format(key, int(item.get("count", 0)))
            for key, item in sorted(self.venues.items())
        )
        return "{} contract(s){} | updated {}".format(
            len(self.contracts),
            " | {}".format(venues) if venues else "",
            self.updated_at or "(never)",
        )


_MAP_CACHE: Dict[str, ContractMap] = {}


def load_contract_map(path: Any = None) -> ContractMap:
    """The map for ``path`` (or the packaged file), cached per resolved path."""
    target = contract_file(path)
    key = str(target)
    cached = _MAP_CACHE.get(key)
    if cached is None:
        cached = ContractMap.load(target)
        _MAP_CACHE[key] = cached
    return cached


def clear_contract_map() -> None:
    """Forget the cached file (a re-fetch in the same process must see it)."""
    _MAP_CACHE.clear()


# ----------------------------------------------------------------- snapshot join
#: The exchange suffix a snapshot record's numeric code carries in Wind.
_EXCHANGE_SUFFIX = {"0": ".SH", "1": ".SZ", "F": ".CFE"}


def record_key(record: Any) -> str:
    """The Wind code of a snapshot record: its code plus the exchange suffix.

    ``('10012493', '0')`` -> ``10012493.SH``; ``('MO2610-C-7000', 'F')`` ->
    ``MO2610-C-7000.CFE``.  A code that already carries a suffix is left alone.
    """
    code = str(getattr(record, "resp_stk_code", "") or getattr(record, "ticker", "") or "")
    code = code.strip().upper()
    if "." in code:
        bare, _, suffix = code.partition(".")
        if not bare:  # a ticker of nothing but a suffix ("*.SH" from an empty code)
            return ""
        return code
    if not code:
        return ""
    suffix = _EXCHANGE_SUFFIX.get(str(getattr(record, "resp_exch_id", "")).strip().upper())
    return "{}{}".format(code, suffix) if suffix else code


def spec_for_record(record: Any, *, path: Any = None) -> Optional[ParsedListedOption]:
    """The terms of a snapshot record, read from the contract file.

    ``None`` means "not in the map" - the caller falls back to parsing the code
    (which works for CFFEX, and used to be the only route for ETF options).
    """
    key = record_key(record)
    if not key:
        return None
    return load_contract_map(path).spec_for_key(key)


__all__ = [
    "DATA_DIR",
    "DEFAULT_DB_HOST",
    "DEFAULT_DB_NAME",
    "DEFAULT_DB_PORT",
    "DEFAULT_DB_USER",
    "FILE_NAME",
    "ORACLE_LIB_CANDIDATES",
    "ContractMap",
    "clear_contract_map",
    "connect",
    "contract_file",
    "fetch_chain",
    "load_contract_map",
    "record_key",
    "spec_for_record",
    "wind_db_url",
    "wind_venue_code",
]
