"""Source-path normalization is shared within an ID-disambiguation pass."""

from pathlib import Path

from graphify.extractors import resolution


def test_repeated_node_edge_and_raw_call_paths_are_resolved_once(monkeypatch, tmp_path: Path):
    first = str(tmp_path / "a.hs")
    second = str(tmp_path / "b.hs")
    nodes = [
        {"id": "shared", "label": "item", "source_file": first},
        {"id": "shared", "label": "item", "source_file": second},
    ]
    edges = [
        {"source": "shared", "target": "shared", "source_file": first}
        for _ in range(40)
    ]
    raw_calls = [
        {"caller_nid": "shared", "callee": "item", "source_file": first}
        for _ in range(100)
    ]
    original = resolution._source_key
    calls: list[str] = []

    def counted(source_file: str, root: Path) -> str:
        calls.append(source_file)
        return original(source_file, root)

    monkeypatch.setattr(resolution, "_source_key", counted)
    resolution._disambiguate_colliding_node_ids(
        nodes, edges, raw_calls, tmp_path.resolve()
    )

    assert calls.count(first) == 1
    assert calls.count(second) == 1
    assert len(calls) == 2
    assert all(raw["caller_nid"] == nodes[0]["id"] for raw in raw_calls)
