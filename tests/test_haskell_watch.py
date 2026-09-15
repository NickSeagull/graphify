"""Incremental graph updates preserve Haskell namespaces and Cabal boundaries."""
import json
from pathlib import Path

import pytest

from graphify.watch import _batch_needs_llm_flag, _batch_triggers_rebuild, _rebuild_code


def _targets(root, caller):
    result = json.loads((root / "graphify-out/graph.json").read_text())
    nodes = {node["id"]: node for node in result["nodes"]}
    return [(edge["relation"], nodes[edge["target"]])
            for edge in result.get("links", result.get("edges", []))
            if nodes[edge["source"]]["label"] == caller]


def test_incremental_watch_keeps_unchanged_haskell_resolution_context(tmp_path, monkeypatch):
    pytest.importorskip("tree_sitter_haskell")
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "Model.hs"
    model.write_text("module Model where\ndata Model = Model Int\n")
    main = tmp_path / "Main.hs"
    main.write_text("module Main where\nimport Model qualified as M\nrun = M.Model 1\n")
    assert _rebuild_code(tmp_path, no_cluster=True)
    main.write_text(
        "module Main where\nimport Model qualified as M\n"
        "run :: M.Model\nrun = M.Model 2\n")
    assert _rebuild_code(tmp_path, changed_paths=[main], no_cluster=True)
    targets = _targets(tmp_path, "run")
    assert any(relation == "calls" and target.get("node_kind") == "constructor"
               and target["source_file"] == "Model.hs" for relation, target in targets)
    assert any(relation == "references" and target.get("node_kind") == "type"
               and target["source_file"] == "Model.hs" for relation, target in targets)


def test_cabal_edit_refreshes_cached_haskell_without_llm(tmp_path, monkeypatch):
    pytest.importorskip("tree_sitter_haskell")
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "demo.cabal"
    prefix = ("name: demo\nlibrary\n  hs-source-dirs: lib\n"
              "  exposed-modules: Api\nexecutable app\n  hs-source-dirs: app\n")
    manifest.write_text(prefix + "  build-depends: demo\n")
    for dirname in ("lib", "app"):
        (tmp_path / dirname).mkdir()
    (tmp_path / "lib/Api.hs").write_text("module Api where\nhelper x = x\n")
    (tmp_path / "app/Main.hs").write_text(
        "module Main where\nimport Api\nrun x = helper x\n")
    assert _rebuild_code(tmp_path, no_cluster=True)
    assert any(relation == "calls" and target["source_file"] == "lib/Api.hs"
               for relation, target in _targets(tmp_path, "run"))
    manifest.write_text(prefix)
    assert _batch_triggers_rebuild([manifest])
    assert not _batch_needs_llm_flag([manifest])
    assert _rebuild_code(tmp_path, changed_paths=[manifest], no_cluster=True)
    targets = [target for relation, target in _targets(tmp_path, "run") if relation == "calls"]
    assert targets and all(not target["source_file"] for target in targets)
