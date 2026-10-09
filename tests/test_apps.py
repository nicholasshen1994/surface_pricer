"""The ``apps/`` contract: every entry point bootstraps itself, and the launcher
dispatches to modules that actually exist.

These checks used to live in ``test_price_tool.py`` next to the one app they were
written for.  With the term-sheet quoting apps gone (2026-10: everything prices
through ``build-json`` -> ``price-json`` / ``slide``) they belong to the folder,
not to a command - a new app cannot be added without them, and a deleted one
cannot leave a dangling dispatch behind.
"""

import re
from pathlib import Path

import pytest

APPS = Path(__file__).resolve().parents[1] / "apps"
#: Modules that are support code, not commands.
HELPERS = {"__init__.py", "_common.py", "_market.py"}


def _app_modules():
    return sorted(path.name for path in APPS.glob("*.py") if path.name not in HELPERS)


def test_the_folder_is_not_empty():
    assert _app_modules()


@pytest.mark.parametrize("name", _app_modules())
def test_every_app_bootstraps_itself(name):
    """``python surface_pricer/apps/<name>.py`` must re-enter through the package.

    The IDE "Run" button launches the file directly, with no package context, so
    each entry point carries the ``__package__`` guard that puts the repository
    root on ``sys.path``.  ``main`` is what a command has to expose.
    """
    text = (APPS / name).read_text(encoding="utf-8")

    assert "__package__ in" in text, "{} cannot be launched directly".format(name)
    assert "parents[2]" in text, "{} must add the repository root".format(name)
    assert "def main(" in text, "{} exposes no main()".format(name)


@pytest.mark.parametrize("name", _app_modules())
def test_an_app_reads_sys_only_with_a_module_level_import(name):
    """An app must not depend on the ``__package__`` guard for ``import sys``.

    Running the file directly re-imports it as a package member, and *that* copy
    never takes the guard branch - so an app that reads ``sys`` has to import it at
    module level.  2026-10: launching ``apps/build_borrow_curve.py`` raised
    ``NameError: name 'sys' is not defined`` exactly this way, through
    ``_no_cli_arguments()``.
    """
    import importlib

    text = (APPS / name).read_text(encoding="utf-8")
    if "sys." not in text:
        return
    assert "import sys" in text, "{} reads sys without importing it".format(name)

    module = importlib.import_module("surface_pricer.apps.{}".format(name[:-3]))
    no_cli = getattr(module, "_no_cli_arguments", None)
    if no_cli is not None:
        assert no_cli() in (True, False)  # a NameError would fail right here


def test_the_launcher_imports_only_existing_modules():
    """Every ``from .apps.X import main`` in the launcher must have a file.

    This is the check that was missing when two app modules were deleted while
    the dispatch (and the package ``__init__``) still named them: ``import
    surface_pricer.apps`` broke for everything, not just for those commands.
    """
    from surface_pricer import __main__ as launcher

    text = Path(launcher.__file__).read_text(encoding="utf-8")
    names = set(re.findall(r"from \.apps\.(\w+) import", text))

    assert names, "the launcher imports no apps at all - did the pattern change?"
    missing = sorted(name for name in names if not (APPS / (name + ".py")).is_file())
    assert not missing, "the launcher dispatches to missing modules: {}".format(missing)


def test_the_retired_quoting_commands_are_gone(capsys):
    """2026-10: ``price-tool`` / ``autocall`` were removed.

    Their capability lives on: ``build-json`` resolves terms into a payload and
    ``price-json`` / ``slide`` price it.  The names must not come back as
    half-restored commands, so the launcher refuses them as unknown.
    """
    from surface_pricer.__main__ import main as launcher_main

    for command in (
        "price-tool",
        "price_tool",
        "autocall",
        "snowball",
        "price",
        "price-trades",
    ):
        assert launcher_main([command]) == 2
    assert "unknown command" in capsys.readouterr().out


def test_build_borrow_curve_picks_its_underlying_from_the_block():
    """The same quick block as ``price-json``: no arguments = the block's venue."""
    from surface_pricer.apps.build_borrow_curve import (
        QUICK_DEFAULTS,
        _apply_quick_defaults,
        _parse_args,
    )

    args = _parse_args([])
    # every key in the block is a real flag dest - a typo must not pass silently
    assert not [key for key in QUICK_DEFAULTS if not hasattr(args, key)]
    applied = _apply_quick_defaults(args)
    assert applied.underlying == QUICK_DEFAULTS["underlying"]
    assert applied.ir_curve == QUICK_DEFAULTS["ir_curve"]
    assert applied.output_root == QUICK_DEFAULTS["output_root"]

    # ... and a mistyped key is refused, with the hint, instead of doing nothing
    QUICK_DEFAULTS["output-root"] = None
    try:
        with pytest.raises(ValueError, match="output_root"):
            _apply_quick_defaults(_parse_args([]))
    finally:
        del QUICK_DEFAULTS["output-root"]


def test_the_launcher_hands_the_block_to_build_borrow_curve(monkeypatch):
    """No arguments -> the block is on; any argument -> the usual CLI rules apply."""
    from surface_pricer.__main__ import main as launcher_main
    from surface_pricer.apps import build_borrow_curve

    calls = []

    def _fake(rest, **kwargs):
        calls.append((list(rest), dict(kwargs)))
        return 0

    monkeypatch.setattr(build_borrow_curve, "main", _fake)

    assert launcher_main(["build-borrow-curve"]) == 0
    assert calls == [([], {"quick": True})]

    calls.clear()
    assert launcher_main(["build-borrow-curve", "--underlying", "IO"]) == 0
    assert calls == [(["--underlying", "IO"], {"quick": False})]


def test_price_json_is_the_pricing_entry_point():
    """The surviving pricing path: build-json -> price-json / slide."""
    from surface_pricer.apps.price_json import QUICK_DEFAULTS, main

    assert callable(main)
    # the block is the whole CLI surface: a handful of the load-bearing keys
    for key in ("payload", "fit", "greeks", "method", "slide", "csv", "json"):
        assert key in QUICK_DEFAULTS
