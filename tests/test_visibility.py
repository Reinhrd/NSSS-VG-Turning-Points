from nsss_vg.graph.visibility import horizontal_visibility_graph


def test_horizontal_visibility_has_nodes():
    values = [1, 3, 2, 5, 4]
    graph = horizontal_visibility_graph(values)
    assert graph.number_of_nodes() == len(values)
    assert graph.number_of_edges() >= len(values) - 1
