"""Removing secrets from the embedding-function configs Chroma persisted.

Chroma stores each collection's embedding-function config (including the load
kwargs the old sentence-transformers wrapper saved, an HF token among them) in
``chroma.sqlite3`` ``collections.config_json_str`` / ``schema_str``.
"""

import json
import sqlite3

import pytest

from backend.utils.chroma_config_scrubber import ChromaConfigScrubber, scrub_json

SECRET = "hf_" + "x" * 34  # never a real token


def _ef(kwargs: dict) -> dict:
    return {
        "type": "known",
        "name": "sentence_transformer",
        "config": {"model_name": "m", "kwargs": kwargs},
    }


def _schema(kwargs: dict) -> dict:
    vector_index = {"config": {"embedding_function": _ef(kwargs), "space": "cosine"}}
    return {
        "defaults": {"float_list": {"vector_index": vector_index}},
        "keys": {"#embedding": {"float_list": {"vector_index": vector_index}}},
    }


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "chroma.sqlite3"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, "
        "config_json_str TEXT, schema_str TEXT)"
    )
    rows = [
        (
            "1",
            "leaky",
            json.dumps({"embedding_function": _ef({"token": SECRET})}),
            json.dumps(_schema({"token": SECRET, "trust_remote_code": False})),
        ),
        ("2", "clean", json.dumps({"hnsw": {"space": "cosine"}}), json.dumps(_schema({}))),
        ("3", "no_schema", None, None),
        # a "token" key outside any embedding-function config is not a secret here
        ("4", "other_token", json.dumps({"metadata": {"token": "keep-me"}}), None),
    ]
    con.executemany("INSERT INTO collections VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


def _row(path, name):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT config_json_str, schema_str FROM collections WHERE name = ?", (name,)
        ).fetchone()
    finally:
        con.close()


class TestScrubJson:
    def test_removes_secrets_inside_embedding_function_configs_only(self):
        text = json.dumps(
            {
                "embedding_function": _ef({"token": SECRET, "device": "mps"}),
                "metadata": {"token": "keep-me"},
            }
        )
        cleaned, removed = scrub_json(text, ("token",))
        data = json.loads(cleaned)
        assert removed == 1
        assert data["embedding_function"]["config"]["kwargs"] == {"device": "mps"}
        assert data["metadata"] == {"token": "keep-me"}

    def test_secret_key_match_ignores_case(self):
        cleaned, removed = scrub_json(
            json.dumps({"embedding_function": {"HF_Token": SECRET}}), ("hf_token",)
        )
        assert removed == 1 and SECRET not in cleaned

    def test_unchanged_text_is_returned_as_is(self):
        text = json.dumps(_schema({}))
        assert scrub_json(text, ("token",)) == (text, 0)

    def test_non_json_is_left_alone(self):
        assert scrub_json("not json", ("token",)) == ("not json", 0)


class TestScrubber:
    def test_scan_reports_without_writing(self, store):
        before = store.read_bytes()

        report = ChromaConfigScrubber(store).scan()

        assert store.read_bytes() == before
        assert report.affected_collections == ["leaky"]
        assert report.secrets_removed == 3  # config_json_str + both schema copies

    def test_apply_removes_every_secret_and_keeps_the_rest(self, store):
        report = ChromaConfigScrubber(store).apply(backup=False)

        assert report.secrets_removed == 3
        config, schema = _row(store, "leaky")
        assert SECRET not in config and SECRET not in schema
        kwargs = json.loads(schema)["defaults"]["float_list"]["vector_index"]["config"][
            "embedding_function"
        ]["config"]["kwargs"]
        assert kwargs == {"trust_remote_code": False}
        assert _row(store, "other_token")[0] == json.dumps({"metadata": {"token": "keep-me"}})
        assert _row(store, "no_schema") == (None, None)

    def test_apply_is_idempotent(self, store):
        scrubber = ChromaConfigScrubber(store)
        scrubber.apply(backup=False)
        assert scrubber.apply(backup=False).secrets_removed == 0

    def test_apply_backs_up_the_store_first(self, store):
        report = ChromaConfigScrubber(store).apply(backup=True)

        assert report.backup_path is not None and report.backup_path.exists()
        con = sqlite3.connect(report.backup_path)
        try:
            leaked = con.execute(
                "SELECT schema_str FROM collections WHERE name = 'leaky'"
            ).fetchone()
        finally:
            con.close()
        assert SECRET in leaked[0]  # the backup is the untouched original

    def test_reports_never_contain_secret_values(self, store):
        report = ChromaConfigScrubber(store).apply(backup=False)
        assert SECRET not in repr(report) and SECRET not in str(report.summary())

    def test_refuses_a_locked_store(self, store):
        holder = sqlite3.connect(store)
        holder.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(RuntimeError, match="locked"):
                ChromaConfigScrubber(store, timeout_seconds=0).apply(backup=False)
        finally:
            holder.rollback()
            holder.close()
        assert SECRET in _row(store, "leaky")[1]

    def test_missing_store_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ChromaConfigScrubber(tmp_path / "absent.sqlite3").scan()
