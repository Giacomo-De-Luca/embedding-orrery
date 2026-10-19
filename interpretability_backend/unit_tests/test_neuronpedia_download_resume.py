"""Neuronpedia downloads stay resumable after an interrupted (cancelled) run.

A cancelled SAE preparation kills the download child mid-write. Batch files are
written then renamed, so resume never skips a truncated batch; the merged
features JSONL drops a half-written last line before appending, so new records
never land on the same line as a torn one.
"""

from interpret.download.download_neuronpedia_s3 import drop_torn_tail, write_atomic


class TestWriteAtomic:
    def test_writes_the_final_name_only(self, tmp_path):
        target = tmp_path / "batch-0.jsonl.gz"

        write_atomic(target, b"payload")

        assert target.read_bytes() == b"payload"
        assert [p.name for p in tmp_path.iterdir()] == ["batch-0.jsonl.gz"]


class TestDropTornTail:
    def test_keeps_complete_lines(self, tmp_path):
        path = tmp_path / "f.jsonl"
        path.write_text('{"index": 0}\n{"index": 1}\n')

        drop_torn_tail(path)

        assert path.read_text() == '{"index": 0}\n{"index": 1}\n'

    def test_cuts_a_half_written_last_line(self, tmp_path):
        path = tmp_path / "f.jsonl"
        path.write_text('{"index": 0}\n{"index": 1}\n{"ind')

        drop_torn_tail(path)

        assert path.read_text() == '{"index": 0}\n{"index": 1}\n'

    def test_single_torn_line_empties_the_file(self, tmp_path):
        path = tmp_path / "f.jsonl"
        path.write_text('{"ind')

        drop_torn_tail(path)

        assert path.read_text() == ""

    def test_finds_the_newline_across_read_chunks(self, tmp_path):
        path = tmp_path / "f.jsonl"
        path.write_text('{"index": 0}\n' + "x" * 200_000)

        drop_torn_tail(path)

        assert path.read_text() == '{"index": 0}\n'
