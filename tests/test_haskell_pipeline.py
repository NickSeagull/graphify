"""Rich Haskell fragments survive old caches, warm caches, and relocation."""
import json
import shutil

import pytest

from graphify.cache import save_cached
from graphify.extract import extract


pytestmark = pytest.mark.usefixtures("_parser")


@pytest.fixture
def _parser():
    pytest.importorskip("tree_sitter_haskell")


def test_legacy_haskell_cache_is_refreshed(tmp_path):
    path = tmp_path / "Main.hs"
    path.write_text("module Main where\nrun x = x\n")
    save_cached(path, {"nodes": [{"id": "old", "label": "old",
                                 "file_type": "code", "source_file": str(path)}],
                       "edges": []}, root=tmp_path, cache_root=tmp_path)
    result = extract([path], root=tmp_path, cache_root=tmp_path, parallel=False)
    assert "old" not in {node["id"] for node in result["nodes"]}
    assert any(node["label"] == "run" and node.get("node_kind")
               for node in result["nodes"])


def test_type_references_survive_warm_and_relocated_cache(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "Model.hs").write_text("module Model where\ndata Model = Model Int\n")
    (original / "Main.hs").write_text(
        "module Main where\nimport Model (Model)\nrun :: Model -> Model\nrun x = x\n")

    def read(root):
        return extract(sorted(root.glob("*.hs")), root=root, cache_root=root, parallel=False)

    cold, warm = read(original), read(original)
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    relocated = read(moved)
    for result in (cold, warm, relocated):
        nodes = {node["id"]: node for node in result["nodes"]}
        assert all(edge["source"] in nodes and edge["target"] in nodes
                   for edge in result["edges"])
        assert any(edge["relation"] == "references"
                   and nodes[edge["source"]]["label"] == "run"
                   and nodes[edge["target"]]["source_file"] == "Model.hs"
                   and not nodes[edge["target"]].get("_callable")
                   for edge in result["edges"])
        assert str(tmp_path) not in json.dumps(result)
    def facts(result):
        return {(edge["source"], edge["target"], edge["relation"])
                for edge in result["edges"]}
    assert facts(cold) == facts(warm) == facts(relocated)
