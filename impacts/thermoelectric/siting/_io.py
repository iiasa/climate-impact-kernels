from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


def refuse_existing_outputs(paths: Iterable[str | Path]) -> None:
    """Raise ``FileExistsError`` naming every path in ``paths`` that already exists.

    Writers call this before they write anything, so a refused write leaves the
    directory as it was and an earlier output is never replaced by accident.
    """

    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output (pass overwrite=True to replace it): "
            + ", ".join(existing)
        )


@contextmanager
def staged_outputs(
    paths: Iterable[str | Path], *, require_all: bool = True
) -> Iterator[dict[Path, Path]]:
    """Write every output of one writer under a hidden sibling name, then move them all into place together.

    Yields a map from each final path to the hidden temporary path the caller
    writes instead. When the block ends cleanly every temporary file replaces its
    final name; on any failure the temporaries are removed and every existing
    output is left as it was. The temporary keeps the final suffix so writers
    that infer the format from it behave the same. The parent directories are
    created. The final step moves each existing output aside first and puts
    them back if any replacement fails. With ``require_all`` false a staged file the caller never wrote is
    skipped instead of raising.
    """

    finals = [Path(path) for path in paths]
    staged = {final: _sibling(final, "tmp") for final in finals}
    for final in finals:
        final.parent.mkdir(parents=True, exist_ok=True)
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        yield staged
        missing = [str(final) for final, temporary in staged.items() if not temporary.exists()]
        if missing and require_all:
            raise RuntimeError("staged output was not written: " + ", ".join(missing))
        ready = [final for final, temporary in staged.items() if temporary.exists()]
        for final in ready:
            if final.exists():
                backups[final] = _sibling(final, "bak")
                final.replace(backups[final])
        for final in ready:
            staged[final].replace(final)
            published.append(final)
    except BaseException:
        for final in published:
            final.unlink(missing_ok=True)
        for final, backup in backups.items():
            if backup.exists():
                backup.replace(final)
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def _sibling(final: Path, kind: str) -> Path:
    return final.with_name(f".{final.stem}.{os.getpid()}.{kind}{final.suffix}")
