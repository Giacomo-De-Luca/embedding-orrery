"""Copies that the copying process can always write to.

``shutil.copy2`` / ``shutil.copytree`` carry the source's permission bits over.
huggingface_hub >= 1.33 keeps xet-backed downloads read-only (0o444) in its
shared blob store, so a mode-preserving copy of a downloaded seed snapshot
leaves the live DuckDB and Chroma files read-only: DuckDB then refuses to open
("Permission denied") and Chroma's sqlite rejects every write.
"""

import shutil
import stat
from pathlib import Path


class WritableCopy:
    """Copy files as new files with the process's default mode (umask applies).

    An existing destination file is replaced, not overwritten in place, so a
    read-only leftover never keeps its mode. Symlinks are followed (Hub cache
    snapshots link into the blob store).
    """

    @staticmethod
    def file(source: Path, destination: Path) -> None:
        Path(destination).unlink(missing_ok=True)
        shutil.copyfile(source, destination)

    @classmethod
    def tree(cls, source: Path, destination: Path, *, dirs_exist_ok: bool = False) -> None:
        try:
            shutil.copytree(
                source, destination, copy_function=cls.file, dirs_exist_ok=dirs_exist_ok
            )
        finally:
            # copytree copies each directory's mode, even when it then raises;
            # the owner must be able to create files in them (sqlite journals,
            # new HNSW segments, a retried copy).
            for directory, _, _ in Path(destination).walk():
                mode = stat.S_IMODE(directory.stat().st_mode)
                if mode & stat.S_IRWXU != stat.S_IRWXU:
                    directory.chmod(mode | stat.S_IRWXU)
