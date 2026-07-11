"""Unit tests for the shared evaluation runner helpers."""

import json

from evaluation.utils import load_config, resolve_config_path, write_results


def test_resolve_config_path_prefers_env(monkeypatch, tmp_path):
    default = tmp_path / "default.toml"
    override = tmp_path / "override.toml"
    monkeypatch.setenv("ORRERY_TEST_CFG", str(override))
    assert resolve_config_path("ORRERY_TEST_CFG", default) == override


def test_resolve_config_path_falls_back_to_default(monkeypatch, tmp_path):
    default = tmp_path / "default.toml"
    monkeypatch.delenv("ORRERY_TEST_CFG", raising=False)
    assert resolve_config_path("ORRERY_TEST_CFG", default) == default


def test_load_config_roundtrips_toml(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('collections = ["a", "b"]\nk = 5\n')
    loaded = load_config(cfg)
    assert loaded == {"collections": ["a", "b"], "k": 5}


def test_write_results_writes_json(tmp_path, capsys):
    out = tmp_path / "results.json"
    results = [{"collection_name": "x", "score": 1.0}]
    write_results(out, results)
    assert json.loads(out.read_text()) == results
    assert "Wrote 1 result(s)" in capsys.readouterr().out
