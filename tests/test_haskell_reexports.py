"""Regression coverage for Haskell module re-exports through shared aliases."""

from pathlib import Path

import pytest

from graphify.extract import extract


pytestmark = pytest.mark.usefixtures("_haskell_parser")


@pytest.fixture
def _haskell_parser():
    pytest.importorskip("tree_sitter_haskell")


def _extract(tmp_path: Path, files: dict[str, str]) -> dict:
    paths = []
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        paths.append(path)
    return extract(
        paths,
        root=tmp_path,
        cache_root=tmp_path / ".cache",
        parallel=False,
    )


def _targets(result: dict, caller: str) -> list[dict]:
    nodes = {node["id"]: node for node in result["nodes"]}
    return [
        nodes[edge["target"]]
        for edge in result["edges"]
        if edge["relation"] == "calls"
        and nodes[edge["source"]]["label"] == caller
    ]


def _assert_target(result: dict, caller: str, label: str, source_file: str) -> None:
    assert any(
        target["label"] == label and target["source_file"] == source_file
        for target in _targets(result, caller)
    )


def test_shared_alias_reexports_every_imported_module(tmp_path):
    result = _extract(tmp_path, {
        "Text.hs": "module Text (trim) where\ntrim x = x\n",
        "List.hs": "module List (first) where\nfirst x = x\n",
        "Core.hs": (
            "module Core (module Public) where\n"
            "import Text as Public\n"
            "import List as Public\n"
        ),
        "Main.hs": (
            "module Main where\n"
            "import Core as C\n"
            "qualifiedCall x = C.trim x\n"
            "unqualifiedCall x = first x\n"
        ),
    })

    _assert_target(result, "qualifiedCall", "trim", "Text.hs")
    _assert_target(result, "unqualifiedCall", "first", "List.hs")


def test_shared_alias_keeps_per_import_lists_hiding_and_children(tmp_path):
    result = _extract(tmp_path, {
        "Result.hs": (
            "module Result (Result(..), internal) where\n"
            "data Result = Ok Int | Err\n"
            "internal x = x\n"
        ),
        "Render.hs": (
            "module Render (Render(..), secret) where\n"
            "class Render value where\n"
            "  render :: value -> String\n"
            "secret x = x\n"
        ),
        "Core.hs": (
            "module Core (module Public) where\n"
            "import Result as Public (Result(..))\n"
            "import Render as Public hiding (secret)\n"
        ),
        "Main.hs": (
            "module Main where\n"
            "import Core\n"
            "make = Ok 1\n"
            "showIt x = render x\n"
            "hiddenOne x = internal x\n"
            "hiddenTwo x = secret x\n"
        ),
    })

    _assert_target(result, "make", "Ok", "Result.hs")
    _assert_target(result, "showIt", "render", "Render.hs")
    assert all(target["source_file"] != "Result.hs" for target in _targets(result, "hiddenOne"))
    assert all(target["source_file"] != "Render.hs" for target in _targets(result, "hiddenTwo"))


def test_nested_shared_reexports_terminate_across_cycles(tmp_path):
    result = _extract(tmp_path, {
        "Leaf.hs": "module Leaf (leaf) where\nleaf x = x\n",
        "Middle.hs": (
            "module Middle (module Shared) where\n"
            "import Leaf as Shared\n"
            "import Cycle as Shared\n"
        ),
        "Cycle.hs": (
            "module Cycle (module Shared) where\n"
            "import Middle as Shared\n"
        ),
        "Top.hs": (
            "module Top (module API) where\n"
            "import Middle as API\n"
        ),
        "Main.hs": "module Main where\nimport Top\nrun x = leaf x\n",
    })

    _assert_target(result, "run", "leaf", "Leaf.hs")


def test_distinct_shared_reexport_targets_remain_ambiguous_despite_proximity(tmp_path):
    result = _extract(tmp_path, {
        "near/Left.hs": "module Left (parse) where\nparse x = x\n",
        "far/deep/Right.hs": "module Right (parse) where\nparse x = x\n",
        "near/Core.hs": (
            "module Core (module Public) where\n"
            "import Left as Public\n"
            "import Right as Public\n"
        ),
        "near/Main.hs": "module Main where\nimport Core\nrun x = parse x\n",
    })

    targets = _targets(result, "run")
    assert len(targets) == 1
    assert targets[0]["label"] in {"parse", "Core.parse"}
    assert targets[0]["source_file"] == ""
    assert targets[0].get("_haskell_unresolved") is True


def test_multiple_reexport_paths_to_same_symbol_are_not_ambiguous(tmp_path):
    result = _extract(tmp_path, {
        "Leaf.hs": "module Leaf (shared) where\nshared x = x\n",
        "Left.hs": "module Left (module Leaf) where\nimport Leaf\n",
        "Right.hs": "module Right (module Leaf) where\nimport Leaf\n",
        "Core.hs": (
            "module Core (module Public) where\n"
            "import Left as Public\n"
            "import Right as Public\n"
        ),
        "Main.hs": "module Main where\nimport Core\nrun x = shared x\n",
    })

    targets = _targets(result, "run")
    assert len(targets) == 1
    assert targets[0]["label"] == "shared"
    assert targets[0]["source_file"] == "Leaf.hs"


def test_downstream_qualified_only_and_alias_rules_survive_reexport(tmp_path):
    result = _extract(tmp_path, {
        "Leaf.hs": "module Leaf (leaf) where\nleaf x = x\n",
        "Core.hs": (
            "module Core (module Public) where\n"
            "import Leaf as Public\n"
        ),
        "Qualified.hs": (
            "module Qualified where\n"
            "import qualified Core as C\n"
            "qualifiedUse x = C.leaf x\n"
            "blockedUse x = leaf x\n"
        ),
        "Aliased.hs": (
            "module Aliased where\n"
            "import Core as C\n"
            "aliasUse x = C.leaf x\n"
            "plainUse x = leaf x\n"
        ),
    })

    _assert_target(result, "qualifiedUse", "leaf", "Leaf.hs")
    _assert_target(result, "aliasUse", "leaf", "Leaf.hs")
    _assert_target(result, "plainUse", "leaf", "Leaf.hs")
    assert all(target["source_file"] != "Leaf.hs" for target in _targets(result, "blockedUse"))


def test_reexport_alias_replaces_original_qualifier_and_excludes_qualified_only(tmp_path):
    result = _extract(tmp_path, {
        "One.hs": "module One (one) where\none x = x\n",
        "Two.hs": "module Two (two) where\ntwo x = x\n",
        "Core.hs": (
            "module Core (module Public) where\n"
            "import One as Public\n"
            "import qualified Two as Public\n"
        ),
        "Main.hs": (
            "module Main where\n"
            "import Core\n"
            "kept x = one x\n"
            "wrongQualifier x = One.one x\n"
            "qualifiedOnly x = two x\n"
        ),
    })

    _assert_target(result, "kept", "one", "One.hs")
    assert all(target["source_file"] != "One.hs" for target in _targets(result, "wrongQualifier"))
    assert all(target["source_file"] != "Two.hs" for target in _targets(result, "qualifiedOnly"))


def test_qualified_alias_reexports_entity_also_imported_unqualified(tmp_path):
    result = _extract(tmp_path, {
        "Leaf.hs": "module Leaf (leaf) where\nleaf x = x\n",
        "Core.hs": (
            "module Core (module Public) where\n"
            "import qualified Leaf as Public\n"
            "import Leaf (leaf)\n"
        ),
        "Main.hs": "module Main where\nimport Core\nthroughCore x = leaf x\n",
        "Direct.hs": (
            "module Direct where\n"
            "import qualified Leaf as L\n"
            "good x = L.leaf x\n"
            "bad x = Leaf.leaf x\n"
        ),
    })

    _assert_target(result, "throughCore", "leaf", "Leaf.hs")
    _assert_target(result, "good", "leaf", "Leaf.hs")
    assert all(target["source_file"] != "Leaf.hs" for target in _targets(result, "bad"))


def test_incremental_context_resolves_all_shared_alias_reexports(tmp_path):
    library = _extract(tmp_path, {
        "One.hs": "module One (one) where\none x = x\n",
        "Two.hs": "module Two (two) where\ntwo x = x\n",
        "Core.hs": (
            "module Core (module Public) where\n"
            "import One as Public\n"
            "import Two as Public\n"
        ),
    })
    main = tmp_path / "Main.hs"
    main.write_text(
        "module Main where\n"
        "import Core qualified as C\n"
        "first x = C.one x\n"
        "second x = C.two x\n"
    )

    changed = extract(
        [main],
        root=tmp_path,
        cache_root=tmp_path / ".changed-cache",
        parallel=False,
        resolution_context_nodes=library["nodes"],
        resolution_context_edges=library["edges"],
    )
    combined = {
        "nodes": library["nodes"] + changed["nodes"],
        "edges": changed["edges"],
    }
    _assert_target(combined, "first", "one", "One.hs")
    _assert_target(combined, "second", "two", "Two.hs")
