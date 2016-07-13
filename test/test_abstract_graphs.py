import metis.abstract_graphs

def example_digraph():
    return metis.abstract_graphs.HashDigraph(
        {0: {1, 2, 3},
         1: {2, 3},
         2: {3},
         3: {}})

def example_graph():
    return metis.abstract_graphs.HashGraph(
        {0: {1, 3},
         1: {2, 0},
         2: {1, 3},
         3: {0, 2}})

def test_digraph_abstract_methods():
    """Test graph theoretic methods"""
    graph = example_digraph()

    assert graph.is_consistent()
    assert not graph.is_undirected()
    assert graph.is_weakly_connected()
    assert not graph.is_strongly_connected()

def test_digraph_container_methods():
    """Test container methods"""
    graph = example_digraph()
    assert 0 in graph
    assert sorted(iter(graph)) == range(4)
    assert len(graph) == 4

def test_digraph_accessor_methods():
    """Test vertex data access"""
    graph = example_digraph()
    assert graph[0] is None
    graph[0] = 'zero'
    assert graph[0] is 'zero'
    del graph[0]
    assert graph[0] is None

def test_digraph_edge_methods():
    """Test edge access and modification"""
    graph = example_digraph()
    assert 1 in graph.successors(0)
    assert 0 in graph.predecessors(1)
    assert 2 not in graph.successors(3)
    graph.add_edge(3, 2)
    assert 2 in graph.successors(3)
    graph.remove_edge(3, 2)
    assert 2 not in graph.successors(3)

def test_digraph_cost_methods():
    """Test cost access and modification"""
    graph = example_digraph()
    assert graph.cost(0, 1) == 1.
    graph.set_cost(0, 1, 2.)
    assert graph.cost(0, 1) == 2.
    graph.remove_edge(0, 1)
    assert graph.cost(0, 1) == float('inf')

def test_graph_abstract_methods():
    """Test graph theoretic methods"""
    graph = example_graph()
    assert graph.is_connected()

def test_graph_container_methods():
    """Test container methods"""
    graph = example_graph()
    assert 0 in graph
    assert sorted(iter(graph)) == range(4)
    assert len(graph) == 4

def test_graph_accessor_methods():
    """Test vertex data access"""
    graph = example_graph()
    assert graph[0] is None
    graph[0] = 'zero'
    assert graph[0] is 'zero'
    del graph[0]
    assert graph[0] is None

def test_graph_edge_methods():
    """Test edge access and modification"""
    graph = example_graph()
    assert 1 in graph.neighbors(0)
    assert 0 in graph.neighbors(1)

    assert 2 not in graph.neighbors(0)
    assert 0 not in graph.neighbors(2)
    graph.add_edge(0, 2)
    assert 2 in graph.neighbors(0)
    assert 0 in graph.neighbors(2)
    graph.remove_edge(0, 2)
    assert 2 not in graph.neighbors(0)
    assert 0 not in graph.neighbors(2)

def test_graph_cost_methods():
    """Test cost access and modification"""
    graph = example_graph()
    assert graph.cost(0, 1) == 1.
    assert graph.cost(1, 0) == 1.
    graph.set_cost(0, 1, 2.)
    assert graph.cost(0, 1) == 2.
    assert graph.cost(1, 0) == 2.
    graph.remove_edge(0, 1)
    assert graph.cost(0, 1) == float('inf')
    assert graph.cost(1, 0) == float('inf')

def test_graph_equivalence():
    """Test semantic equivalence"""
    graph1 = example_graph()
    graph2 = example_graph()
    assert graph1 is not graph2

    assert graph1 == graph2
    assert graph2 == graph1
    assert hash(graph1) == hash(graph2)

    graph1.add_edge(0, 2)
    assert graph1 != graph2
    assert graph2 != graph1
    assert hash(graph1) != hash(graph2)

def test_digraph_equivalence():
    """Test semantic equivalence"""
    graph1 = example_digraph()
    graph2 = example_digraph()
    assert graph1 is not graph2

    assert graph1 == graph2
    assert graph2 == graph1
    assert hash(graph1) == hash(graph2)

    graph1.add_edge(3, 2)
    assert graph1 != graph2
    assert graph2 != graph1
    assert hash(graph1) != hash(graph2)
