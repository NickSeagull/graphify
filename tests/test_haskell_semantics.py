"""Haskell declaration metadata and explicit source-level type relationships."""
from pathlib import Path

import pytest

from graphify.extractors.haskell import extract_haskell
from graphify.extract import extract


@pytest.fixture
def parse(tmp_path):
    pytest.importorskip("tree_sitter_haskell")

    def extract_source(source: str):
        path = tmp_path / "Demo.hs"
        path.write_text(source)
        return extract_haskell(path)

    return extract_source


def by_label(result, label):
    return [node for node in result["nodes"] if node["label"] == label]


def reference_facts(result):
    labels = {node["id"]: node["label"] for node in result["nodes"]}
    return {
        (
            labels[ref["caller_nid"]], ref["callee"], ref["relation"],
            ref["namespace"], ref["context"], ref["source_location"],
        )
        for ref in result["raw_references"]
    }


def test_declaration_metadata_preserves_syntax_and_namespace(parse):
    result = parse(
        "module Demo (Limit(..), C(method), run) where\n"
        "-- | A bounded quantity.\n"
        "newtype Limit = Limit { amount :: Int }\n"
        "class C a where\n"
        "  method :: a -> Limit\n"
        "run :: Limit -> Int\n"
        "run (Limit n) = n\n"
    )

    type_node, constructor = by_label(result, "Limit")
    if type_node["node_kind"] != "type":
        type_node, constructor = constructor, type_node
    assert type_node["id"] != constructor["id"]
    assert type_node["node_kind"] == "type"
    assert type_node["haddock_summary"] == "A bounded quantity."
    assert type_node["end_line"] == 3
    assert constructor["node_kind"] == "constructor"
    assert constructor["parent"] == "Limit"
    assert constructor["exported"] is True
    assert by_label(result, "amount")[0]["node_kind"] == "field"

    method = by_label(result, "method")[0]
    assert method["node_kind"] == "function"
    assert method["parent"] == "C"
    assert method["written_signature"] == "method :: a -> Limit"
    run = by_label(result, "run")[0]
    assert run["written_signature"] == "run :: Limit -> Int"
    assert run["language"] == "haskell"
    assert result["_haskell_schema"] == 3


def test_explicit_type_relationships_have_exact_source_context(parse):
    result = parse(
        "module Demo where\n"
        "type Alias a = Either String a\n"
        "class (Eq a, Show a) => C a where\n"
        "  method :: Maybe a -> Alias a\n"
        "data Thing a = Thing { payload :: Maybe a }\n"
        "instance C (Thing Int) where\n"
        "  method = undefined\n"
        "type family F a\n"
        "type instance F (Thing a) = Maybe a\n"
        "run :: (C a, Eq b) => Thing a -> Alias b\n"
        "run = undefined\n"
    )
    facts = reference_facts(result)
    expected = {
        ("Alias", "Either", "aliases", "type", "type_alias_rhs", "L2"),
        ("C", "Eq", "constrains", "type", "class_constraint", "L3"),
        ("method", "Maybe", "references", "type", "written_signature", "L4"),
        ("Thing", "Maybe", "references", "type", "declaration_type", "L5"),
        ("instance C (Thing Int)", "C", "instance_of", "type", "instance_class", "L6"),
        ("instance C (Thing Int)", "Thing", "references", "type", "instance_head", "L6"),
        ("type instance F (Thing a)", "F", "instance_of", "type", "type_family_instance", "L9"),
        ("type instance F (Thing a)", "Maybe", "references", "type", "type_family_rhs", "L9"),
        ("run", "C", "constrains", "type", "signature_constraint", "L10"),
        ("run", "Alias", "references", "type", "written_signature", "L10"),
    }
    assert expected <= facts
    assert all(ref["language"] == "haskell" for ref in result["raw_references"])
    assert not any(ref["callee"] == "Thing" and ref["context"] == "declaration_type"
                   for ref in result["raw_references"])


def test_public_pipeline_preserves_haskell_symbol_metadata(tmp_path):
    pytest.importorskip("tree_sitter_haskell")
    path = tmp_path / "Library.hs"
    path.write_text(
        "module Library (Token) where\n"
        "-- | Public token parser.\n"
        "parseToken :: Token -> Bool\n"
        "parseToken = undefined\n"
    )
    result = extract([path], root=tmp_path, cache_root=tmp_path, parallel=False)
    node = next(item for item in result["nodes"] if item["label"] == "parseToken")
    assert node["node_kind"] == "function"
    assert node["language"] == "haskell"
    assert node["written_signature"] == "parseToken :: Token -> Bool"
    assert node["haddock_summary"] == "Public token parser."
    assert node["exported"] is False
    assert node["end_line"] == 4


def test_qualified_instances_positional_types_closed_families_and_shared_signatures(parse):
    result = parse(
        "module Demo where\n"
        "f, g :: Q.Type -> Bool\n"
        "data Outer = Outer Inner | Pair Inner Q.Other\n"
        "type family Family a where\n"
        "  Family Inner = Outer\n"
        "instance Q.C Inner where\n"
        "  method = undefined\n"
    )
    refs = result["raw_references"]
    labels = {node["id"]: node["label"] for node in result["nodes"]}
    facts = {(labels[ref["caller_nid"]], ref["qualified_prefix"], ref["callee"],
              ref["relation"], ref["context"]) for ref in refs}
    assert {("f", "Q", "Type", "references", "written_signature"),
            ("g", "Q", "Type", "references", "written_signature")} <= facts
    assert {("Outer", None, "Inner", "references", "declaration_type"),
            ("Outer", "Q", "Other", "references", "declaration_type")} <= facts
    assert {("Family", None, "Inner", "references", "type_family_lhs"),
            ("Family", None, "Outer", "references", "type_family_rhs")} <= facts
    instance = next(node for node in result["nodes"] if node["node_kind"] == "instance")
    assert (instance["label"], "Q", "C", "instance_of", "instance_class") in facts
    assert by_label(result, "method")[0]["node_kind"] == "method"


def test_constructor_and_case_folded_record_field_keep_distinct_kinds(parse):
    result = parse(
        "data FieldSchema = FieldSchema { fieldSchema :: Bool }\n"
        "make = FieldSchema { fieldSchema = True }\n"
    )
    matches = by_label(result, "FieldSchema") + by_label(result, "fieldSchema")
    assert {node["node_kind"] for node in matches} == {"type", "constructor", "field"}
    assert len({node["id"] for node in matches}) == 3
