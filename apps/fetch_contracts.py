"""Fetch the listed-option contract terms out of Wind into one local file.

``python -m surface_pricer fetch-contracts --underlying 510500`` reads a venue's
**whole** option chain from ``WINDDF.CHINAOPTIONDESCRIPTION`` and merges it into
``data/option_contracts.json`` (:mod:`surface_pricer.marketdata.option_contracts`).
That file is what the pricing path reads instead of the database: a quote never
waits for Oracle, and a contract only has to be fetched **once** - the merge skips
everything the file already knows, so re-running it is cheap and the file grows
only when a new contract appears.

This command is the only door to the database in the whole tool chain.  Several
venues in one go: ``--underlying 510500,MO``.

Connection: ``WIND_DB_URL`` (``user/password@host:port/service``) or the
``WIND_DB_USER`` / ``WIND_DB_PASSWORD`` / ``WIND_DB_HOST`` / ``WIND_DB_PORT`` /
``WIND_DB_NAME`` variables - put them in ``surface_pricer/.env`` (git-ignored).
The Wind server here is older than python-oracledb's thin mode supports, so an
Instant Client is loaded when needed (``ORACLE_CLIENT_LIB_DIR`` points at it).
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - plain script launch
    # IDE "Run" launches the file with no package context; put the repository root
    # on sys.path so the imports below resolve the same way as under -m.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from typing import Iterable, List, Optional

from ..marketdata.option_contracts import (
    ContractMap,
    contract_file,
    fetch_chain,
    wind_venue_code,
)
from ..marketdata.registry import get_underlying_spec
from ._common import load_env_files


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    load_env_files(args.env_file)

    names: List[str] = [
        item.strip() for item in str(args.underlying or "").split(",") if item.strip()
    ]
    if not names:
        print("ERROR: --underlying needs at least one venue, e.g. --underlying 510500")
        return 2
    try:
        specs = [get_underlying_spec(name) for name in names]
    except KeyError as error:
        print("ERROR: {}".format(error))
        return 2

    contract_map = ContractMap.load(args.file)
    print("file       : {} | {}".format(contract_file(args.file), contract_map.describe()))

    for spec in specs:
        try:
            entries = fetch_chain(spec, url=args.db_url)
        except Exception as error:  # noqa: BLE001 - the driver's errors all mean "no fetch"
            print("ERROR: cannot fetch {}: {}".format(spec.underlying, error))
            return 2
        added, total = contract_map.merge(spec, entries)
        print(
            "{:8s} | chain {:14s} | fetched {:5d} | new {:4d} | total {:5d}".format(
                spec.underlying,
                wind_venue_code(spec),
                len(entries),
                added,
                total,
            )
        )

    target = contract_file(args.file)
    try:
        contract_map.save(args.file)
    except OSError as error:
        print("ERROR: cannot write {}: {}".format(target, error))
        return 2
    print("written    : {}".format(target))
    return 0


def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="surface_pricer fetch-contracts",
        description=(
            "fetch listed-option contract terms from Wind into "
            "data/option_contracts.json (the file the pricing path reads)"
        ),
    )
    parser.add_argument(
        "--underlying",
        required=True,
        help="venue(s) to fetch: one name or a comma list, e.g. '510500' or '510500,MO'",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="contract file to update (default: surface_pricer/data/option_contracts.json)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "Wind connection string user/password@host:port/service "
            "(default: WIND_DB_URL / WIND_DB_* variables)"
        ),
    )
    parser.add_argument("--env-file", default=None, help="additional .env with WIND_DB_*")
    return parser.parse_args(list(argv) if argv is not None else None)


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - plain script launch
    raise SystemExit(main())
