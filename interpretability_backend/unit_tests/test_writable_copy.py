"""Unit tests for WritableCopy (copies that never inherit a read-only source mode)."""

import stat

from backend.utils.writable_copy import WritableCopy

READ_ONLY_FILE = 0o444  # like huggingface_hub >= 1.33 xet blobs
READ_ONLY_DIR = stat.S_IRUSR | stat.S_IXUSR


def _owner_writable(path) -> bool:
    # Mode bit, not os.access: root may write to anything.
    return bool(path.stat().st_mode & stat.S_IWUSR)


def test_file_copy_is_writable_when_source_is_read_only(tmp_path):
    source = tmp_path / "blob"
    source.write_bytes(b"payload")
    source.chmod(READ_ONLY_FILE)
    destination = tmp_path / "copy"

    WritableCopy.file(source, destination)

    assert destination.read_bytes() == b"payload"
    assert _owner_writable(destination)


def test_file_copy_replaces_a_read_only_destination(tmp_path):
    source = tmp_path / "blob"
    source.write_bytes(b"new")
    destination = tmp_path / "main.duckdb.seedtmp"
    destination.write_bytes(b"stale")
    destination.chmod(READ_ONLY_FILE)

    WritableCopy.file(source, destination)

    assert destination.read_bytes() == b"new"
    assert _owner_writable(destination)


def test_tree_copy_makes_files_and_directories_writable(tmp_path):
    source = tmp_path / "source"
    segment = source / "segment"
    segment.mkdir(parents=True)
    (source / "chroma.sqlite3").write_bytes(b"catalog")
    (segment / "data_level0.bin").write_bytes(b"vectors")
    (source / "chroma.sqlite3").chmod(READ_ONLY_FILE)
    (segment / "data_level0.bin").chmod(READ_ONLY_FILE)
    segment.chmod(READ_ONLY_DIR)
    destination = tmp_path / "destination"

    try:
        WritableCopy.tree(source, destination)
    finally:
        segment.chmod(stat.S_IRWXU)  # lets pytest remove tmp_path

    assert (destination / "segment" / "data_level0.bin").read_bytes() == b"vectors"
    for path in (
        destination,
        destination / "segment",
        destination / "chroma.sqlite3",
        destination / "segment" / "data_level0.bin",
    ):
        assert _owner_writable(path), path


def test_tree_copy_follows_hub_cache_symlinks(tmp_path):
    """Hub snapshots are relative symlinks into a read-only blob store; the copy
    must hold real files (the BuildKit cache mount is not part of the image)."""
    cache = tmp_path / "cache"
    blob = cache / "blobs" / "0d" / "0d0f56aa"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"vectors")
    blob.chmod(READ_ONLY_FILE)
    snapshot = cache / "snapshots" / "rev" / "demo"
    snapshot.mkdir(parents=True)
    (snapshot / "data_level0.bin").symlink_to("../../../blobs/0d/0d0f56aa")
    destination = tmp_path / "staged"

    WritableCopy.tree(snapshot, destination)

    copied = destination / "data_level0.bin"
    assert not copied.is_symlink()
    assert copied.read_bytes() == b"vectors"
    assert _owner_writable(copied)


def test_tree_copy_merges_into_an_existing_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.bin").write_bytes(b"new")
    (source / "chroma.sqlite3").write_bytes(b"seed catalog")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "kept.bin").write_bytes(b"kept")
    (destination / "chroma.sqlite3").write_bytes(b"read-only leftover")
    (destination / "chroma.sqlite3").chmod(READ_ONLY_FILE)

    WritableCopy.tree(source, destination, dirs_exist_ok=True)

    assert (destination / "new.bin").read_bytes() == b"new"
    assert (destination / "kept.bin").read_bytes() == b"kept"
    assert (destination / "chroma.sqlite3").read_bytes() == b"seed catalog"
    assert _owner_writable(destination / "chroma.sqlite3")
