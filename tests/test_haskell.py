"""Haskell grammar and public extraction regression coverage."""
from pathlib import Path
import sys

import pytest

from graphify.detect import classify_file, FileType
from graphify.extract import extract, extract_haskell, _EXTRA_FOR_EXTENSION


def write(tmp_path, source, name="Main.hs"):
    path = tmp_path / name
    path.write_text(source)
    return path


@pytest.fixture
def parse(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    return lambda source: extract_haskell(write(tmp_path, source))


def pairs(result, relation):
    labels = {n["id"]: n["label"] for n in result["nodes"]}
    return {(labels[e["source"]], labels[e["target"]]) for e in result["edges"]
            if e["relation"] == relation}


def deferred_calls(result):
    return {
        (call["callee"], call.get("qualified_prefix"), call.get("context"))
        for call in result.get("raw_calls", [])
    }


def test_definitions_and_imports(parse):
    result = parse('''module Demo.Shapes where
import qualified Data.List as L
import Data.Maybe (fromMaybe)
data Shape = Circle Double | Box { width :: Double, height :: Double }
newtype Name = Name String
type Count = Int
class Render a where
  render :: a -> String
  render x = show x
instance Render Shape where
  render (Circle r) = show r
helper, later :: Int -> Int
helper 0 = 0
helper x = later x
later x = x + 1
run xs = L.map helper xs
''')
    assert "error" not in result
    labels = {n["label"] for n in result["nodes"]}
    assert {"Demo.Shapes", "Shape", "Circle", "Box", "width", "height", "Name",
            "Count", "Render", "render", "helper", "later", "run"} <= labels
    assert ("Main.hs", "Demo.Shapes") in pairs(result, "defines")
    assert ("Demo.Shapes", "Shape") in pairs(result, "contains")
    assert ("Shape", "Circle") in pairs(result, "contains")
    assert ("Render", "render") in pairs(result, "contains")
    assert ("helper", "later") in pairs(result, "calls")
    assert ("Demo.Shapes", "Data.List") in pairs(result, "imports_from")
    assert len([n for n in result["nodes"] if n["label"] == "helper"]) == 1
    renders = [n for n in result["nodes"] if n["label"] == "render"]
    assert len(renders) == 2 and renders[0]["id"] != renders[1]["id"]
    ids = {n["id"] for n in result["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in result["edges"])
    assert all(n["source_location"].startswith("L") for n in result["nodes"] if n["source_file"])
    assert all(n.get("origin_file") and not n["source_location"]
               for n in result["nodes"] if not n["source_file"])


def test_scopes_and_forward_calls(parse):
    result = parse('''helper x = x
run x = local x where
  local y = helper y
other x = let local y = helper y in local x
shadow helper x = helper x
lambdaShadow x = (\\helper -> helper x) id
''')
    calls = pairs(result, "calls")
    assert {("run", "local"), ("other", "local"), ("local", "helper")} <= calls
    assert ("shadow", "helper") not in calls
    assert ("lambdaShadow", "helper") not in calls
    locals_ = [n for n in result["nodes"] if n["label"] == "local"]
    assert len(locals_) == 2
    assert locals_[0]["id"] != locals_[1]["id"]


def test_qualified_calls_and_operators(parse):
    result = parse('''module Main where
import qualified Elsewhere as E
helper x = E.helper x
run x = Main.helper x
x <+> y = helper x
x <*> y = helper y
use x = x <+> 2
backtick x = x `helper` 2
''')
    calls = pairs(result, "calls")
    assert ("helper", "E", "call") in deferred_calls(result)
    assert ("helper", "helper") not in calls
    assert ("run", "helper") in calls
    assert ("use", "<+>") in calls
    assert ("backtick", "helper") in calls
    ops = [n for n in result["nodes"] if n["label"] in {"<+>", "<*>"}]
    assert len(ops) == 2 and ops[0]["id"] != ops[1]["id"]


def test_gadt_and_values(parse):
    result = parse('''{-# LANGUAGE GADTs #-}
data T a where
  MkT :: a -> T a
value = MkT 1
''')
    assert ("T", "MkT") in pairs(result, "contains")
    assert ("value", "MkT") in pairs(result, "calls")


def test_empty_and_missing_file(parse, tmp_path):
    assert len(parse("")["nodes"]) == 1
    result = extract_haskell(tmp_path / "missing.hs")
    assert result["nodes"] == result["edges"] == []
    assert "error" in result


def test_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tree_sitter_haskell", None)
    result = extract_haskell(write(tmp_path, "main = pure ()"))
    assert result == {"nodes": [], "edges": [], "error": "tree-sitter-haskell not installed"}
    assert _EXTRA_FOR_EXTENSION[".hs"] == "haskell"


def test_public_pipeline_and_cross_file_resolution(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    library = write(tmp_path, "module Library where\nhelper x = x\n", "Library.hs")
    main = write(tmp_path, "module Main where\nimport Library\nrun x = helper x\n")
    assert classify_file(main) == FileType.CODE
    result = extract([library, main], root=tmp_path, cache_root=tmp_path, parallel=False)
    assert ("run", "helper") in pairs(result, "calls")
    helpers = [n for n in result["nodes"] if n["label"] == "helper"]
    assert len(helpers) == 1 and helpers[0]["source_file"] == "Library.hs"


def test_pattern_and_sequential_scopes(parse):
    result = parse('''helper x = x
(a,b) = (1,2)
branch x = case x of
  Just helper -> helper 1
  Nothing -> helper 2
sequenceCall x = do
  helper <- pure id
  helper x
comprehension xs = [helper x | helper <- xs, x <- xs]
''')
    calls = pairs(result, "calls")
    assert ("branch", "helper") in calls  # the Nothing branch
    assert ("sequenceCall", "helper") not in calls
    assert ("comprehension", "helper") not in calls
    assert {"a", "b"} <= {n["label"] for n in result["nodes"]}
    labels = {n["id"]: n["label"] for n in result["nodes"]}
    assert not any(e["relation"] == "calls" and labels[e["target"]] == "helper"
                   and e["source_location"] == "L4" for e in result["edges"])


def test_operator_stubs_and_qualified_shadowing(parse):
    result = parse('''module Main where
helper x = x
shadow helper x = Main.helper x
math x = x + 1 * 2
patternOnly x = case x of
  Just y -> y
''')
    calls = pairs(result, "calls")
    assert ("shadow", "helper") in calls
    assert {("+", None, "call"), ("*", None, "call")} <= deferred_calls(result)
    assert ("patternOnly", "Just") not in calls


def test_primed_names_remain_distinct_in_public_pipeline(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    path = write(tmp_path, "helper x = x\nhelper' x = helper x\nrun x = helper' x\n")
    result = extract([path], root=tmp_path, cache_root=tmp_path, parallel=False)
    assert {("run", "helper'"), ("helper'", "helper")} <= pairs(result, "calls")
    helpers = [n for n in result["nodes"] if n["label"] in {"helper", "helper'"}]
    assert len(helpers) == 2 and helpers[0]["id"] != helpers[1]["id"]


def test_import_aware_resolution_disambiguates_common_names(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    left = write(tmp_path, "module Left where\nmap x = x\n", "Left.hs")
    right = write(tmp_path, "module Right where\nmap x = x\n", "Right.hs")
    main = write(
        tmp_path,
        "module Main where\n"
        "import Left qualified as L\n"
        "import Right qualified as R\n"
        "left x = L.map x\n"
        "right x = R.map x\n",
    )
    result = extract(
        [left, right, main], root=tmp_path, cache_root=tmp_path, parallel=False
    )
    labels = {node["id"]: node for node in result["nodes"]}
    calls = [edge for edge in result["edges"] if edge["relation"] == "calls"]
    assert any(
        labels[edge["source"]]["label"] == "left"
        and labels[edge["target"]]["source_file"] == "Left.hs"
        for edge in calls
    )
    assert any(
        labels[edge["source"]]["label"] == "right"
        and labels[edge["target"]]["source_file"] == "Right.hs"
        for edge in calls
    )


def test_explicit_import_hiding_and_external_qualification(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    library = write(
        tmp_path,
        "module Library (visible, hidden) where\nvisible x = x\nhidden x = x\n",
        "Library.hs",
    )
    main = write(
        tmp_path,
        "module Main where\n"
        "import Library hiding (hidden)\n"
        "visibleCall x = visible x\n"
        "hiddenCall x = hidden x\n"
        "external x = External.visible x\n",
    )
    result = extract([library, main], root=tmp_path, cache_root=tmp_path, parallel=False)
    labels = {node["id"]: node for node in result["nodes"]}
    calls = [edge for edge in result["edges"] if edge["relation"] == "calls"]

    def targets(caller):
        return [labels[edge["target"]] for edge in calls
                if labels[edge["source"]]["label"] == caller]

    assert any(node["source_file"] == "Library.hs" for node in targets("visibleCall"))
    assert all(node["source_file"] != "Library.hs" for node in targets("hiddenCall"))
    assert {node["label"] for node in targets("external")} == {"External.visible"}


def test_module_reexport_and_constructor_import(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    model = write(
        tmp_path,
        "module Model (Result(..)) where\ndata Result = Ok Int | Err\n",
        "Model.hs",
    )
    facade = write(
        tmp_path,
        "module Facade (module Model) where\nimport Model\n",
        "Facade.hs",
    )
    main = write(
        tmp_path,
        "module Main where\nimport Facade (Result(..))\nrun = Ok 1\n",
    )
    result = extract(
        [model, facade, main], root=tmp_path, cache_root=tmp_path, parallel=False
    )
    labels = {node["id"]: node for node in result["nodes"]}
    calls = [edge for edge in result["edges"] if edge["relation"] == "calls"]
    assert any(
        labels[edge["source"]]["label"] == "run"
        and labels[edge["target"]]["label"] == "Ok"
        and labels[edge["target"]]["source_file"] == "Model.hs"
        for edge in calls
    )


def test_qualified_constructor_wins_over_same_named_type(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    model = write(
        tmp_path,
        "module Model (Limit(..)) where\nnewtype Limit = Limit Int\n",
        "Model.hs",
    )
    main = write(
        tmp_path,
        "module Main where\nimport Model qualified\nrun = Model.Limit 10\n",
    )
    result = extract(
        [model, main], root=tmp_path, cache_root=tmp_path, parallel=False
    )
    nodes = {node["id"]: node for node in result["nodes"]}
    targets = [nodes[edge["target"]] for edge in result["edges"]
               if edge["relation"] == "calls" and nodes[edge["source"]]["label"] == "run"]
    assert len(targets) == 1
    assert targets[0]["label"] == "Limit"
    assert targets[0].get("_callable") is True


def test_pipe_point_free_splice_and_type_instance(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    library = write(
        tmp_path,
        "module Library where\nfromIO x = x\nderiveThing x = x\n",
        "Library.hs",
    )
    main = write(
        tmp_path,
        "{-# LANGUAGE TemplateHaskell #-}\n"
        "module Main where\n"
        "import Library qualified as L\n"
        "run x = x |> L.fromIO\n"
        "alias = L.fromIO\n"
        "L.deriveThing ''Int\n"
        "type family NameOf a\n"
        "type instance NameOf Int = String\n",
    )
    result = extract([library, main], root=tmp_path, cache_root=tmp_path, parallel=False)
    calls = pairs(result, "calls")
    assert {("run", "fromIO"), ("alias", "fromIO"), ("Main", "deriveThing")} <= calls
    assert "type instance NameOf Int" in {node["label"] for node in result["nodes"]}
    contexts = {edge.get("context") for edge in result["edges"] if edge["relation"] == "calls"}
    assert {"pipeline", "point_free_alias"} <= contexts


def test_record_dot_does_not_bind_to_imported_or_local_name(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    library = write(tmp_path, "module Library where\nrun x = x\n", "Library.hs")
    main = write(
        tmp_path,
        "module Main where\nimport Library (run)\nuse record = record.run 1\n",
    )
    result = extract([library, main], root=tmp_path, cache_root=tmp_path, parallel=False)
    labels = {node["id"]: node for node in result["nodes"]}
    targets = [labels[edge["target"]] for edge in result["edges"]
               if edge["relation"] == "calls" and labels[edge["source"]]["label"] == "use"]
    assert {node["label"] for node in targets} == {"record.run"}
    assert all(not node["source_file"] for node in targets)


def test_parse_recovery_metadata_is_returned(parse):
    result = parse("module Main where\nrun = do\n  (-1) |> Int.powerOf 2 |> shouldBe 1\n")
    assert result["parse_errors"]["first_error_line"] == 2
    assert result["parse_errors"]["error_count"] >= 1
    assert result["parse_errors"]["material_recovery"] is True


def test_parse_recovery_is_reported_by_public_pipeline(tmp_path, capsys):
    pytest.importorskip("tree_sitter_haskell")
    path = write(
        tmp_path,
        "module Main where\nrun = do\n  (-1) |> Int.powerOf 2 |> shouldBe 1\n",
    )
    extract([path], root=tmp_path, cache_root=tmp_path, parallel=False)
    warning = capsys.readouterr().err
    assert "had syntax errors" in warning
    assert "Main.hs" in warning


def test_duplicate_modules_resolve_by_nearest_source_tree(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    first_dir = tmp_path / "package-a" / "src"
    second_dir = tmp_path / "package-b" / "src"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = write(first_dir, "module App where\nstart x = x\n", "App.hs")
    second = write(second_dir, "module App where\nstart x = x\n", "App.hs")
    caller = write(
        tmp_path / "package-a",
        "module Runner where\nimport App qualified\nrun x = App.start x\n",
        "Runner.hs",
    )
    result = extract(
        [first, second, caller], root=tmp_path, cache_root=tmp_path, parallel=False
    )
    nodes = {node["id"]: node for node in result["nodes"]}
    targets = [nodes[edge["target"]] for edge in result["edges"]
               if edge["relation"] == "calls" and nodes[edge["source"]]["label"] == "run"]
    assert len(targets) == 1
    assert targets[0]["source_file"] == "package-a/src/App.hs"


def test_incremental_resolution_uses_unchanged_haskell_context(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    library = write(tmp_path, "module Library where\nhelper x = x\n", "Library.hs")
    first = extract(
        [library], root=tmp_path, cache_root=tmp_path / "first", parallel=False
    )
    main = write(
        tmp_path,
        "module Main where\nimport Library qualified as L\nrun x = L.helper x\n",
    )
    changed = extract(
        [main],
        root=tmp_path,
        cache_root=tmp_path / "second",
        parallel=False,
        resolution_context_nodes=first["nodes"],
        resolution_context_edges=first["edges"],
    )
    labels = {node["id"]: node for node in first["nodes"] + changed["nodes"]}
    targets = [labels[edge["target"]] for edge in changed["edges"]
               if edge["relation"] == "calls" and labels[edge["source"]]["label"] == "run"]
    assert len(targets) == 1
    assert targets[0]["source_file"] == "Library.hs"


def test_explicit_class_children_are_exported(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    library = write(
        tmp_path,
        "module Render (Render(..)) where\n"
        "class Render value where\n  render :: value -> String\n  render x = show x\n",
        "Render.hs",
    )
    main = write(
        tmp_path,
        "module Main where\nimport Render (Render(..))\nrun x = render x\n",
    )
    result = extract(
        [library, main], root=tmp_path, cache_root=tmp_path, parallel=False
    )
    labels = {node["id"]: node for node in result["nodes"]}
    assert any(
        edge["relation"] == "calls"
        and labels[edge["source"]]["label"] == "run"
        and labels[edge["target"]]["label"] == "render"
        and labels[edge["target"]]["source_file"] == "Render.hs"
        for edge in result["edges"]
    )


def test_class_method_signature_without_default_is_callable(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    library = write(
        tmp_path,
        "module Convert (Convert(..)) where\n"
        "class Convert value where\n  convert :: value -> String\n",
        "Convert.hs",
    )
    main = write(
        tmp_path,
        "module Main where\n"
        "import Convert qualified\n"
        "run x = Convert.convert x\n",
    )
    result = extract(
        [library, main], root=tmp_path, cache_root=tmp_path, parallel=False
    )
    labels = {node["id"]: node for node in result["nodes"]}
    targets = [
        labels[edge["target"]]
        for edge in result["edges"]
        if edge["relation"] == "calls"
        and labels[edge["source"]]["label"] == "run"
    ]
    assert len(targets) == 1
    assert targets[0]["label"] == "convert"
    assert targets[0]["source_file"] == "Convert.hs"


def test_record_construction_calls_constructor_but_update_does_not(parse):
    result = parse(
        "data Record = Record { value :: Int }\n"
        "make = Record { value = 1 }\n"
        "update old = old { value = 2 }\n"
    )
    labels = {node["id"]: node["label"] for node in result["nodes"]}
    calls = [edge for edge in result["edges"] if edge["relation"] == "calls"]
    assert any(
        labels[edge["source"]] == "make"
        and labels[edge["target"]] == "Record"
        and edge.get("context") == "record_construction"
        for edge in calls
    )
    assert not any(
        labels[edge["source"]] == "update"
        and labels[edge["target"]] == "Record"
        for edge in calls
    )


def test_repeated_names_in_separate_local_scopes_stay_distinct(parse):
    result = parse(
        "run x =\n"
        "  ( let local y = y in local x\n"
        "  , let local y = y + 1 in local x\n"
        "  )\n"
    )
    locals_ = [node for node in result["nodes"] if node["label"] == "local"]
    assert len(locals_) == 2
    assert locals_[0]["id"] != locals_[1]["id"]
    targets = {
        edge["target"]
        for edge in result["edges"]
        if edge["relation"] == "calls"
        and edge["target"] in {node["id"] for node in locals_}
    }
    assert targets == {node["id"] for node in locals_}


def test_pattern_synonyms_and_foreign_imports_are_callable(parse):
    result = parse(
        "{-# LANGUAGE PatternSynonyms #-}\n"
        "pattern Nil = []\n"
        "foreign import ccall \"sin\" c_sin :: Double -> Double\n"
        "empty = Nil\n"
        "sine x = c_sin x\n"
    )
    assert {"Nil", "c_sin"} <= {node["label"] for node in result["nodes"]}
    assert {("empty", "Nil"), ("sine", "c_sin")} <= pairs(result, "calls")
