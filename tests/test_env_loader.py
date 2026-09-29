from surface_pricer.apps._common import load_env_files, parse_env_file


def test_parse_env_file_sets_only_missing_variables(tmp_path, monkeypatch):
    monkeypatch.delenv("SURFACE_PRICER_TEST_KEY", raising=False)
    monkeypatch.setenv("SURFACE_PRICER_TEST_EXISTING", "keep")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "SURFACE_PRICER_TEST_KEY='secret value'\n"
        "SURFACE_PRICER_TEST_EXISTING=override\n"
        "MALFORMED_LINE\n",
        encoding="utf-8",
    )

    parse_env_file(env_file)

    import os

    assert os.environ["SURFACE_PRICER_TEST_KEY"] == "secret value"
    assert os.environ["SURFACE_PRICER_TEST_EXISTING"] == "keep"


def test_load_env_files_tolerates_missing_files(tmp_path):
    load_env_files(str(tmp_path / "does-not-exist.env"))


def test_load_env_files_reads_an_explicit_file(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("SURFACE_PRICER_EXPLICIT_KEY", raising=False)
    env_file = tmp_path / "custom.env"
    env_file.write_text("SURFACE_PRICER_EXPLICIT_KEY=from-explicit\n", encoding="utf-8")

    load_env_files(str(env_file))

    assert os.environ["SURFACE_PRICER_EXPLICIT_KEY"] == "from-explicit"
