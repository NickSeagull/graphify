"""Named Haskell re-exports and namespace-aware reference resolution."""

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
    return extract(paths, root=tmp_path, cache_root=tmp_path / ".cache", parallel=False)


def _edges(result: dict, relation: str, source_label: str) -> list[dict]:
    nodes = {node["id"]: node for node in result["nodes"]}
    return [
        nodes[edge["target"]]
        for edge in result["edges"]
        if edge["relation"] == relation
        and nodes[edge["source"]]["label"] == source_label
    ]


def test_named_type_and_children_reexport_through_nested_facades(tmp_path):
    result = _extract(tmp_path, {
        "Model.hs": (
            "module Model (Position(..), hidden) where\n"
            "data Position = Position Int | Origin\n"
            "hidden = Origin\n"
        ),
        "Facade.hs": (
            "module Facade (Position(..)) where\n"
            "import Model (Position(..), hidden)\n"
        ),
        "Public.hs": (
            "module Public (Position(..)) where\n"
            "import Facade (Position(..))\n"
        ),
        "Main.hs": (
            "module Main where\n"
            "import Public\n"
            "origin :: Position\n"
            "origin = Origin\n"
            "bad = hidden\n"
        ),
    })

    type_targets = _edges(result, "references", "origin")
    assert any(node["label"] == "Position" and node["node_kind"] == "type"
               and node["source_file"] == "Model.hs" for node in type_targets)
    call_targets = _edges(result, "calls", "origin")
    assert any(node["label"] == "Origin" and node["node_kind"] == "constructor"
               and node["source_file"] == "Model.hs" for node in call_targets)
    assert all(node.get("source_file") != "Model.hs"
               for node in _edges(result, "calls", "bad"))


def test_same_written_type_and_constructor_resolve_by_namespace(tmp_path):
    result = _extract(tmp_path, {
        "Model.hs": "module Model (Limit(..)) where\nnewtype Limit = Limit Int\n",
        "Facade.hs": "module Facade (Limit(..)) where\nimport Model (Limit(..))\n",
        "Main.hs": (
            "module Main where\n"
            "import qualified Facade as F\n"
            "make :: F.Limit\n"
            "make = F.Limit 10\n"
        ),
    })

    type_targets = _edges(result, "references", "make")
    call_targets = _edges(result, "calls", "make")
    assert any(node["node_kind"] == "type" and node["source_file"] == "Model.hs"
               for node in type_targets)
    assert any(node["node_kind"] == "constructor" and node["source_file"] == "Model.hs"
               for node in call_targets)


def test_qualified_named_type_and_children_are_reexported(tmp_path):
    result = _extract(tmp_path, {
        "Model.hs": "module Model (Limit(..)) where\nnewtype Limit = Limit Int\n",
        "Facade.hs": (
            "module Facade (F.Limit(..)) where\n"
            "import qualified Model as F\n"
        ),
        "Main.hs": (
            "module Main where\n"
            "import Facade\n"
            "make :: Limit\n"
            "make = Limit 10\n"
        ),
    })

    assert any(node.get("node_kind") == "type" and node.get("source_file") == "Model.hs"
               for node in _edges(result, "references", "make"))
    assert any(node.get("node_kind") == "constructor"
               and node.get("source_file") == "Model.hs"
               for node in _edges(result, "calls", "make"))


def test_bare_type_import_does_not_import_same_named_constructor(tmp_path):
    result = _extract(tmp_path, {
        "Model.hs": "module Model (Limit(..)) where\nnewtype Limit = Limit Int\n",
        "TypesOnly.hs": (
            "module TypesOnly where\n"
            "import Model (Limit)\n"
            "typed :: Limit\n"
            "typed = Limit 1\n"
        ),
        "WithChildren.hs": (
            "module WithChildren where\n"
            "import Model (Limit(..))\n"
            "built = Limit 2\n"
        ),
    })

    assert any(node.get("node_kind") == "type" and node.get("source_file") == "Model.hs"
               for node in _edges(result, "references", "typed"))
    assert all(node.get("source_file") != "Model.hs"
               for node in _edges(result, "calls", "typed"))
    assert any(node.get("node_kind") == "constructor"
               and node.get("source_file") == "Model.hs"
               for node in _edges(result, "calls", "built"))


def test_import_filter_blocks_child_from_explicit_reexport(tmp_path):
    result = _extract(tmp_path, {
        "Model.hs": "module Model (Choice(..)) where\ndata Choice = Yes | No\n",
        "Facade.hs": (
            "module Facade (Choice(..)) where\n"
            "import Model (Choice(Yes))\n"
        ),
        "Main.hs": (
            "module Main where\n"
            "import Facade\n"
            "good = Yes\n"
            "bad = No\n"
        ),
    })

    assert any(node.get("source_file") == "Model.hs"
               for node in _edges(result, "calls", "good"))
    assert all(node.get("source_file") != "Model.hs"
               for node in _edges(result, "calls", "bad"))


def test_cabal_visibility_is_checked_at_each_facade_hop(tmp_path):
    (tmp_path / "demo.cabal").write_text(
        "name: demo\n"
        "version: 0.1\n"
        "library\n"
        "  hs-source-dirs: src\n"
        "  exposed-modules: Facade\n"
        "  other-modules: Leaf Private\n"
        "  build-depends: base\n"
        "executable demo-app\n"
        "  hs-source-dirs: app\n"
        "  main-is: Main.hs\n"
        "  build-depends: base, demo\n"
    )
    result = _extract(tmp_path, {
        "src/Leaf.hs": "module Leaf (leaf) where\nleaf x = x\n",
        "src/Private.hs": "module Private (secret) where\nsecret x = x\n",
        "src/Facade.hs": (
            "module Facade (leaf) where\n"
            "import Leaf (leaf)\n"
        ),
        "app/Main.hs": (
            "module Main where\n"
            "import Facade (leaf)\n"
            "import Private (secret)\n"
            "through x = leaf x\n"
            "blocked x = secret x\n"
        ),
    })

    assert any(node.get("source_file") == "src/Leaf.hs"
               for node in _edges(result, "calls", "through"))
    assert all(node.get("source_file") != "src/Private.hs"
               for node in _edges(result, "calls", "blocked"))


def test_conditional_dependency_does_not_certify_resolution(tmp_path):
    (tmp_path / "demo.cabal").write_text(
        "name: demo\n"
        "version: 0.1\n"
        "library\n"
        "  hs-source-dirs: src\n"
        "  exposed-modules: Facade\n"
        "  build-depends: base\n"
        "executable demo-app\n"
        "  hs-source-dirs: app\n"
        "  main-is: Main.hs\n"
        "  build-depends: base\n"
        "  if flag(use-demo)\n"
        "    build-depends: demo\n"
    )
    result = _extract(tmp_path, {
        "src/Facade.hs": "module Facade (api) where\napi x = x\n",
        "app/Main.hs": (
            "module Main where\n"
            "import Facade (api)\n"
            "run x = api x\n"
        ),
    })

    targets = _edges(result, "calls", "run")
    assert len(targets) == 1
    assert targets[0].get("_haskell_unresolved") is True


def test_own_module_qualified_type_reference_resolves_locally(tmp_path):
    result = _extract(tmp_path, {
        "Main.hs": (
            "module Main where\n"
            "data T = T\n"
            "run :: Main.T\n"
            "run = T\n"
        ),
    })

    targets = _edges(result, "references", "run")
    assert any(node.get("label") == "T" and node.get("node_kind") == "type"
               and node.get("source_file") == "Main.hs" for node in targets)
    assert all(node.get("_haskell_unresolved") is not True for node in targets)


def test_named_reexport_diamond_resolves_one_leaf_definition(tmp_path):
    result = _extract(tmp_path, {
        "Leaf.hs": "module Leaf (shared) where\nshared x = x\n",
        "Left.hs": "module Left (shared) where\nimport Leaf (shared)\n",
        "Right.hs": "module Right (shared) where\nimport Leaf (shared)\n",
        "Top.hs": (
            "module Top (shared) where\n"
            "import Left (shared)\n"
            "import Right (shared)\n"
        ),
        "Main.hs": "module Main where\nimport Top (shared)\nrun x = shared x\n",
    })

    targets = _edges(result, "calls", "run")
    assert len(targets) == 1
    assert targets[0].get("label") == "shared"
    assert targets[0].get("source_file") == "Leaf.hs"


def test_no_field_selectors_keeps_record_field_out_of_call_namespace(tmp_path):
    (tmp_path / "demo.cabal").write_text(
        "name: demo\n"
        "version: 0.1\n"
        "common options\n"
        "  default-extensions: NoFieldSelectors\n"
        "library\n"
        "  import: options\n"
        "  hs-source-dirs: src\n"
        "  exposed-modules: Client, Types\n"
        "  build-depends: base\n"
        "executable app\n"
        "  import: options\n"
        "  hs-source-dirs: app\n"
        "  main-is: Main.hs\n"
        "  build-depends: base, demo\n"
    )
    result = _extract(tmp_path, {
        "src/Types.hs": (
            "module Types (TokenSet(..)) where\n"
            "data TokenSet = TokenSet { refreshToken :: Int }\n"
        ),
        "src/Client.hs": (
            "module Client (refreshToken) where\n"
            "import Types (TokenSet(..))\n"
            "refreshToken x = x\n"
        ),
        "app/Main.hs": (
            "module Main where\n"
            "import qualified Client as OAuth2\n"
            "run x = OAuth2.refreshToken x\n"
        ),
    })

    targets = _edges(result, "calls", "run")
    assert len(targets) == 1
    assert targets[0].get("node_kind") == "function"
    assert targets[0].get("source_file") == "src/Client.hs"


def test_source_field_selectors_pragma_overrides_cabal_default(tmp_path):
    (tmp_path / "demo.cabal").write_text(
        "name: demo\n"
        "version: 0.1\n"
        "library\n"
        "  default-extensions: NoFieldSelectors\n"
        "  hs-source-dirs: src\n"
        "  exposed-modules: Types\n"
        "  build-depends: base\n"
        "executable app\n"
        "  hs-source-dirs: app\n"
        "  main-is: Main.hs\n"
        "  build-depends: base, demo\n"
    )
    result = _extract(tmp_path, {
        "src/Types.hs": (
            "{-# LANGUAGE FieldSelectors #-}\n"
            "module Types (TokenSet(..)) where\n"
            "data TokenSet = TokenSet { refreshToken :: Int }\n"
        ),
        "app/Main.hs": (
            "module Main where\n"
            "import Types (TokenSet(..))\n"
            "run tokenSet = refreshToken tokenSet\n"
        ),
    })

    targets = _edges(result, "calls", "run")
    assert len(targets) == 1
    assert targets[0].get("node_kind") == "field"
    assert targets[0].get("source_file") == "src/Types.hs"
