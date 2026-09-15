import networkx as nx

from graphify.serve import (
    _node_detail_text, _query_graph_text, _score_nodes, _subgraph_to_text,
)


def _graph():
    graph = nx.Graph()
    graph.add_node(
        "type-id", label="StreamId", norm_label="streamid", language="haskell",
        node_kind="type", written_signature="newtype StreamId = StreamId Text",
        haddock_summary="Stable stream identity", package="nhcore",
        components=["library:nhcore"], end_line=14, source_file="src/StreamId.hs",
        source_location="L10", _haskell_cabal={"private": "must-not-render"},
    )
    graph.add_node(
        "constructor-id", label="StreamId", norm_label="streamid", language="haskell",
        node_kind="constructor", package="nhcore", components=["library:nhcore"],
        source_file="src/StreamId.hs", source_location="L10",
    )
    return graph


def test_haskell_kind_and_signature_disambiguate_search_ranking():
    ranked = _score_nodes(_graph(), ["StreamId", "type"])
    assert ranked[0][1] == "type-id"


def test_query_text_renders_public_haskell_metadata_without_private_context():
    text = _query_graph_text(_graph(), "StreamId type", depth=0)
    assert "kind=type" in text
    assert "signature=newtype StreamId = StreamId Text" in text
    assert "doc=Stable stream identity" in text
    assert "package=nhcore" in text
    assert "components=library:nhcore" in text
    assert "end=L14" in text
    assert "_haskell_cabal" not in text
    assert "must-not-render" not in text


def test_get_node_detail_renders_same_public_haskell_metadata():
    text = _node_detail_text(_graph(), "type-id")
    assert "Kind: type" in text
    assert "Signature: newtype StreamId = StreamId Text" in text
    assert "Doc: Stable stream identity" in text
    assert "Package: nhcore" in text
    assert "Components: library:nhcore" in text
    assert "must-not-render" not in text


def test_non_haskell_output_and_ranking_are_unchanged_by_haskell_fields():
    graph = nx.Graph()
    graph.add_node("plain", label="Thing", source_file="thing.py", source_location="L1")
    graph.add_node(
        "ignored", label="Other", language="python", node_kind="Thing",
        written_signature="Thing", source_file="other.py", source_location="L1",
    )
    assert _score_nodes(graph, ["Thing"])[0][1] == "plain"
    text = _subgraph_to_text(graph, {"plain"}, [], seeds=["plain"])
    assert "kind=" not in text
    assert "signature=" not in text
