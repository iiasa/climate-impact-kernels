from __future__ import annotations

from pathlib import Path

import pytest

from impacts.thermoelectric.siting._io import staged_outputs


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def test_a_failed_publish_restores_every_earlier_output(tmp_path, monkeypatch) -> None:
    finals = [tmp_path / f"f{i}.txt" for i in range(3)]
    for final in finals:
        final.write_text("old " + final.name)
    before = _snapshot(tmp_path)

    real = Path.replace
    publishes = []

    def flaky(self, target):
        if ".tmp" in self.name:
            publishes.append(self.name)
            if len(publishes) == 2:
                raise OSError("rename failed")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    with pytest.raises(OSError, match="rename failed"), staged_outputs(finals) as staged:
        for final in finals:
            staged[final].write_text("new " + final.name)

    assert _snapshot(tmp_path) == before
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_a_successful_publish_leaves_only_the_new_outputs(tmp_path) -> None:
    finals = [tmp_path / "a.txt", tmp_path / "b.txt"]
    finals[0].write_text("old")
    with staged_outputs(finals) as staged:
        for final in finals:
            staged[final].write_text("new")

    assert _snapshot(tmp_path) == {"a.txt": b"new", "b.txt": b"new"}
