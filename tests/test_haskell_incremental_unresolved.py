"""Incremental Haskell resolution keeps newly created unresolved targets."""

from pathlib import Path

import pytest

from graphify.extract import extract


def test_incremental_unresolved_target_survives_import_stub_pruning(tmp_path: Path):
    pytest.importorskip("tree_sitter_haskell")
    library = tmp_path / "Library.hs"
    library.write_text("module Library (helper) where\nhelper x = x\n")
    unchanged = extract(
        [library], root=tmp_path, cache_root=tmp_path / ".first", parallel=False
    )

    main = tmp_path / "Main.hs"
    main.write_text(
        "module Main where\n"
        "import qualified Library as L\n"
        "run x = L.missing x\n"
    )
    changed = extract(
        [main],
        root=tmp_path,
        cache_root=tmp_path / ".second",
        parallel=False,
        resolution_context_nodes=unchanged["nodes"],
        resolution_context_edges=unchanged["edges"],
    )

    nodes = {node["id"]: node for node in changed["nodes"]}
    run_id = next(node["id"] for node in changed["nodes"] if node["label"] == "run")
    call = next(
        edge for edge in changed["edges"]
        if edge["relation"] == "calls" and edge["source"] == run_id
    )
    target = nodes[call["target"]]
    assert target["label"] == "Library.missing"
    assert target.get("_haskell_unresolved") is True
