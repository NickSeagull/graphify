"""Haskell utility fan-in influences grouping without altering query facts."""
from copy import deepcopy

import networkx as nx
import pytest

from graphify.cluster import _haskell_clustering_projection, cluster


def _graph(*, distinct_files=True, multigraph=False, language="haskell"):
    graph = nx.MultiDiGraph() if multigraph else nx.DiGraph()
    suffix = "hs" if language == "haskell" else "py"
    graph.add_node("pipe", language=language, source_file=f"Basics.{suffix}")
    graph.add_node("type", language=language, source_file=f"Model.{suffix}")
    for i in range(12):
        graph.add_node(str(i), source_file=f"Caller{i if distinct_files else 0}.{suffix}")
        graph.add_edge(str(i), "pipe", relation="calls", weight=2.0)
        graph.add_edge(str(i), "type", relation="references", weight=1.0)
    return graph


@pytest.mark.parametrize("multigraph", [False, True])
def test_haskell_partition_projection_preserves_query_graph(multigraph):
    graph = _graph(multigraph=multigraph)
    before = deepcopy(graph)
    projection = _haskell_clustering_projection(graph)
    assert nx.utils.graphs_equal(graph, before)
    assert set(projection.nodes) == set(graph.nodes)
    assert projection.number_of_edges() == graph.number_of_edges()
    for _, _, attrs in projection.edges(data=True):
        if attrs["relation"] == "calls":
            assert 0 < attrs["weight"] < 1.0
        else:
            assert attrs["weight"] == 1.0


def test_repeated_calls_from_one_file_do_not_make_a_global_helper():
    graph = _graph(distinct_files=False)
    assert nx.utils.graphs_equal(_haskell_clustering_projection(graph), graph)


def test_other_languages_keep_their_existing_partition_weights():
    graph = _graph(language="python")
    assert nx.utils.graphs_equal(_haskell_clustering_projection(graph), graph)


def test_default_build_preserves_direction_for_haskell_projection():
    from graphify.build import build_from_json

    directed = _graph()
    extraction = {
        "nodes": [{"id": n, "label": n, **a} for n, a in directed.nodes(data=True)],
        "edges": [{"source": s, "target": t, **a} for s, t, a in directed.edges(data=True)],
    }
    graph = build_from_json(extraction)
    assert not graph.is_directed()
    before = deepcopy(graph)
    projected = _haskell_clustering_projection(graph)
    assert projected["0"]["pipe"]["weight"] < graph["0"]["pipe"]["weight"]
    assert projected["0"]["type"]["weight"] == 1.0
    assert nx.utils.graphs_equal(graph, before)


def test_unoriented_undirected_graph_is_not_guessed():
    graph = _graph().to_undirected()
    assert _haskell_clustering_projection(graph) is graph


def test_partition_uses_projection_and_can_reproduce_legacy_weights(monkeypatch):
    import graphify.cluster as module
    graph = _graph()
    seen = []

    def partition(projected, resolution=1.0):
        seen.append(projected["0"]["pipe"]["weight"])
        return {node: index for index, node in enumerate(projected.nodes)}

    monkeypatch.setattr(module, "_partition", partition)
    assert {n for group in cluster(graph).values() for n in group} == set(graph)
    cluster(graph, downweight_haskell_hubs=False)
    assert seen[0] < seen[1] == 2.0
    assert graph["0"]["pipe"]["weight"] == 2.0
