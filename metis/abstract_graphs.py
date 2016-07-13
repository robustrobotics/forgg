"""Implements graph abstractions

Conceptually, a graph is just a tuple (V, E), with V a set of vertex
labels and E a set of pairs of vertex labels. Two vertices $u, v \in E$
are connected if and only if $(u, v) \in E$.  If the graph is directed,
the pairs in E are ordered; if it is undirected, the pairs are
unordered.

However, it is rarely sensible to represent a graph as a list of
vertices and a list of edges. The obvious pracical implementation is
with a lookup mapping each vertex to its neighbors (in python, that is
easy to represent with a dict), but that requires the graph to be
represented explicitly in memory. That is problematic if the number of
vertices is exponentially large (say, if the set of vertices is the set
product of smaller discrete sets), or if the edges are calculated by a
subroutine (say, if the edges are between all pairs of points closer
than some fixed distance).

We support such implicit graphs by defining an abstraction, which allows
for clear and efficient representation of directed and undirected graphs
without requiring them to be explicitly calculated ahead of time.
"""

class AbstractDigraph(object):
    """Abstract definition of a directed graph

    Fundamentally, a directed graph need implement only a successor
    method, which returns all vertices connected to a given vertex by a
    directed edge. A predecessor method is also often useful, and this
    abstraction assumes one is implemented, although it is possible to
    avoid that and this class may be rewritten in the future to avoid
    the additional restriction.

    As an abstraction, this class has no attributes and stores no data.
    It exists purely to define an interface and implement some common
    functions (like checking connectivity) in terms of that interface.
    """
    def __contains__(self, vertex):
        """Check if vertex is the label of a vertex in the graph"""
        raise NotImplementedError

    def __iter__(self):
        """Iterate over the labels of vertices in the graph"""
        raise NotImplementedError

    def __len__(self):
        """Return the cardinality of the set of vertices"""
        raise NotImplementedError

    def __getitem__(self, label):
        """Return the vertex data associated with the label"""
        raise NotImplementedError

    def __setitem__(self, label, data):
        """Set the vertex data associated with the label"""
        raise NotImplementedError

    def __delitem__(self, label):
        """Remove a vertex from the graph"""
        raise NotImplementedError

    def __eq__(self, other):
        """Check for semantic equivalence of graphs

        Two graphs are equivalent iff they have the same vertex labels,
        vertex data, and edges, regardless of how they are represented.
        """
        my_labels = set(self)
        other_labels = set(self)
        if my_labels != other_labels:
            return False
        for vertex in self:
            if self[vertex] != other[vertex]:
                return False
            if set(self.successors(vertex)) != set(other.successors(vertex)):
                return False
            if set(self.predecessors(vertex)) != set(other.predecessors(vertex)):
                return False
        return True

    def __neq__(self, other):
        """Check for semantic non-equivalence of graphs

        Two graphs are non-equivalent only if they have the same vertex labels,
        vertex data, or edges, regardless of how they are represented.
        """
        return not self.__eq__(other)

    def __hash__(self):
        """Hash function that respects semantic equivalence"""
        return hash(tuple(
            (vertex, self[vertex],) +
            tuple(sorted(self.successors(vertex))) +
            tuple(sorted(self.predecessors(vertex)))
            for vertex in sorted(iter(self))))

    def predecessors(self, vertex):
        """Return the vertices with vertex as a child

        A vertex `v` is a predecessor of a vertex `u` if the ordered
        pair `(v, u)` is an edge in the graph. This method returns the
        predecessors of `vertex` as a sequence.

        Args:
            vertex: the label of a vertex in the graph

        Returns:
            a sequence of vertices with `vertex` as a parent. No
                guarantee is provided about the form of this sequence,
                other than that it will support iteration; often, the
                sequence will be an iterator or a set.

        Raises:
            ValueError if `vertex` is not the label of a vertex in the
            graph
        """
        raise NotImplementedError

    def successors(self, vertex):
        """Return the vertices with vertex as a parent

        A vertex `v` is a successor of a vertex `u` if the ordered pair
        `(u, v)` is an edge in the graph. This method returns the
        successors of `vertex` as a sequence.

        Args:
            vertex: the label of a vertex in the graph

        Returns:
            a sequence of vertices with `vertex` as a parent. No
                guarantee is provided about the form of this sequence,
                other than that it will support iteration; often, the
                sequence will be an iterator or a set.

        Raises:
            ValueError if `vertex` is not the label of a vertex in the
            graph
        """
        raise NotImplementedError

    def add_edge(self, parent, child, cost=None):
        """Add an edge from parent to child

        If either vertex is not in the graph, it will be added.
        """
        raise NotImplementedError

    def remove_edge(self, parent, child):
        """Remove the edge from parent to child, if it exists"""
        raise NotImplementedError

    def cost(self, parent, child):
        """Return the cost associated with an edge

        Note the default behavior (which can be overridden) is to return
        the unit cost for any edge which exists, and an infinite cost
        for any edge which does not.

        Args:
            parent: the label of the start of the edge
            child: the label of the end of the edge

        Returns:
            float: the cost associated with the edge. If the edge does
                not exist, the cost is infinite. However, the cost may
                be infinite even for edges generated with `neighbors`;
                for example, an implementation of an implicit geometric
                graph will defer expensive collision detection
                operations until the cost is queried.

        Raises:
            ValueError if `parent` or `child` is not the label of a
            vertex in the graph
        """
        if child in self.successors(parent):
            return 1.
        else:
            return float('inf')

    def set_cost(self, parent, child, cost):
        """Set the cost of an edge"""
        raise NotImplementedError

    def is_consistent(self):
        """Check if a directed graph is consistent

        A graph is consistent if adjacency is reflexive: that is, if for
        every vertex pair `u, v`, `v in graph.successors(u)` implies `u
        in graph.predecessors(v)`. This method iterates over each vertex
        to see if that property holds.

        Returns:
            bool: True if the graph is consistent
        """
        for vertex in self:
            for successor in self.successors(vertex):
                if vertex not in self.predecessors(successor):
                    return False
        return True

    def is_undirected(self, allow_self_edge=False):
        """Check if a graph is undirected

        A graph is undirected if for every vertex pair `u, v`, `v in
        graph.successors(u)` implies `u in graph.successors(v)`. This method
        iterates over each vertex to see if that property holds.

        Args:
            allow_self_edge (bool): By convention, undirected graphs are
                typically assumed to not have edges connecting a vertex
                to itself. If allow_self_edge is False, then any graph
                with `u in graph.successors(u)` will be considered not
                undirected.  If true, self-edges will be ignored when
                determining if a graph is undirected.

        Returns:
            bool: True if the graph is undirected

        Examples:
            >>> HashGraph({0: {1, 2}, 1: {0}, 2: {0}}).is_undirected()
            True
            >>> HashGraph({0: {1, 2}, 1: {0}, 2: {}}).is_undirected()
            False
        """
        seen = set()
        needed = set()
        for vertex in self:
            if not allow_self_edge and vertex in self.successors(vertex):
                return False
            else:
                for successor in self.successors(vertex):
                    if self.cost(vertex, successor) < float('inf'):
                        seen.add((vertex, successor))
                        needed.discard((vertex, successor))
                        if (successor, vertex) not in seen:
                            needed.add((successor, vertex))
        return len(needed) == 0

    def is_weakly_connected(self):
        """Check if a graph is weakly connected

        A graph is weakly connected if there is an undirected path
        between every pair of vertices. That is, a graph is weakly
        connected if it would be connected if it were made undirected by
        adding the edge (v, u) for each edge (u, v).

        Returns:
            bool: True if the graph is weakly connected

        Examples:
            >>> d1 = {0: {1, 2}, 1: {}, 2: {}}
            >>> HashDigraph(d1).is_weakly_connected()
            True
            >>> d2 = {0: {1}, 1: {}, 2: {}}
            >>> HashDigraph(d2).is_weakly_connected()
            False
        """
        seen = set() # Vertices seen so far
        queue = set() # Vertices not yet seen

        # initial set includes the first vertex in iterator order
        queue.add(next(iter(self)))
        while queue:
            current = queue.pop()
            seen.add(current)
            neighbors = {v for v in self.successors(current)
                         if self.cost(current, v) < float('inf')}
            neighbors |= {v for v in self.predecessors(current)
                          if self.cost(v, current) < float('inf')}
            queue |= self.successors(current) - seen
        return len(seen) == len(self)

    def strongly_connected_components(self):
        """Return a list of strongly connected components

        A subset of vertices of a graph is strongly connected if there
        is a path between every pair of vertices in the subset, such
        that for each sequential pair (u, v) of vertices on the path, `v
        in graph.successors(u)` is `True`.

        This function implements Tarjan's algorithm to obtain the
        strongly connected components, and is largely taken from
        http://www.logarithmic.net/pfh/blog/01208083168

        Returns:
            list(list): each element in the list is a list of vertices
                in a single strongly connected component
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        result = []

        def get_scc(vertex):
            """Recursively compute strongly connected components"""
            # set the depth index for this vertex to the smallest unused index
            index[vertex] = index_counter[0]
            lowlinks[vertex] = index_counter[0]
            index_counter[0] += 1
            stack.append(vertex)

            for successor in self.successors(vertex):
                if self.cost(vertex, successor) < float('inf'):
                    if successor not in lowlinks:
                        # Successor has not yet been visited; recurse on it
                        get_scc(successor)
                        lowlinks[vertex] = min(lowlinks[vertex],
                                               lowlinks[successor])
                    elif successor in stack:
                        # the successor is in the stack and hence in the
                        # current strongly connected component
                        lowlinks[vertex] = min(lowlinks[vertex],
                                               index[successor])

            # If `vertex` is a root vertex, pop the stack and generate an SCC
            if lowlinks[vertex] == index[vertex]:
                connected_component = []

                while True:
                    successor = stack.pop()
                    connected_component.append(successor)
                    if successor == vertex:
                        break
                component = tuple(connected_component)
                result.append(component)

        for vertex in self:
            if vertex not in lowlinks:
                get_scc(vertex)

        return result

    def is_strongly_connected(self):
        """Check if a graph is strongly connected

        A graph is strongly connected if there is a path between every
        pair of vertices, such that for each sequential pair (u, v) of
        vertices on the path, `v in graph.successors(u)` is `True`.

        Returns:
            bool: True if the graph is strongly connected

        Examples:
            >>> d1 = {0: {1, 2}, 1: {}, 2: {}}
            >>> HashDigraph(d1).is_strongly_connected()
            False
            >>> d2 = {0: {1}, 1: {2}, 2: {0}}
            >>> HashDigraph(d2).is_strongly_connected()
            True
        """
        return len(self.strongly_connected_components()) == 1

class UndirectedGraph(AbstractDigraph):
    """Abstract definition of a weighted undirected graph

    An undirected graph need implement only a neighbors method, which
    returns all vertices adjacent to a given vertex. An undirected graph
    is also implicitly a directed graph, where each undirected edge
    corresponds to two directed edges; this abstraction therefore
    implements the successors() and predecessors() methods
    appropriately.

    As an abstraction, this class has no attributes and stores no data.
    It exists purely to define an interface and implement some common
    functions (like checking connectivity) in terms of that interface.
    """
    def __eq__(self, other):
        """Check for semantic equivalence of graphs

        Two graphs are equivalent iff they have the same vertex labels,
        vertex data, and edges, regardless of how they are represented.
        """
        my_labels = set(self)
        other_labels = set(self)
        if my_labels != other_labels:
            return False
        for vertex in self:
            if self[vertex] != other[vertex]:
                return False
            if set(self.neighbors(vertex)) != set(other.neighbors(vertex)):
                return False
        return True

    def __neq__(self, other):
        """Check for semantic non-equivalence of graphs

        Two graphs are non-equivalent only if they have the same vertex labels,
        vertex data, or edges, regardless of how they are represented.
        """
        return not self.__eq__(other)

    def __hash__(self):
        """Hash function that respects semantic equivalence"""
        return hash(tuple(
            (vertex, self[vertex]) + tuple(sorted(self.neighbors(vertex)))
            for vertex in sorted(iter(self))))

    def neighbors(self, vertex):
        """Return the vertices with vertex as a neighbor

        Args:
            vertex: the vertex whose neighbors should be queries

        Returns:
            a sequence of vertices neighboring `vertex`. No guarantee is
                provided about the form of this sequence, other than
                that it will support iteration; often, the sequence will
                be an iterator or a set.

        Raises:
            ValueError if `vertex` is not the label of a vertex in the
                graph
        """
        raise NotImplementedError

    def predecessors(self, vertex):
        """Return the vertices with vertex as a neighbor

        Method is defined to provide a uniform interface with
        DirectedGraph; it need not be overridden, since for any
        undirected graph the set of predessors is precisely the set of
        neighbors.

        Args:
            vertex: the vertex whose neighbors should be queries

        Returns:
            a sequence of vertices neighboring `vertex`. No guarantee is
                provided about the form of this sequence, other than
                that it will support iteration; often, the sequence will
                be an iterator or a set.

        Raises:
            ValueError if vertex is not in the graph
        """
        return self.neighbors(vertex)

    def successors(self, vertex):
        """Return the vertices with vertex as a neighbor

        Method is defined to provide a uniform interface with
        DirectedGraph; it need not be overridden, since for any
        undirected graph the set of successors is precisely the set of
        neighbors.

        Args:
            vertex: the vertex whose neighbors should be queries

        Returns:
            a sequence of vertices neighboring `vertex`. No guarantee is
                provided about the form of this sequence, other than
                that it will support iteration; often, the sequence will
                be an iterator or a set.

        Raises:
            ValueError if vertex is not in the graph
        """
        return self.neighbors(vertex)

    def cost(self, parent, child):
        """Return the cost associated with an edge

        By definition, cost(parent, child) = cost(child, parent) in an
        undirected graph; implementations should respect this
        constraint.

        Args:
            parent: the label of the start of the edge
            child: the label of the end of the edge

        Returns:
            float: the cost associated with the edge. If the edge does
                not exist, the cost is infinite. However, the cost may
                be infinite even for edges generated with `neighbors`;
                many implementations of implicit graphs will defer
                expensive computation of edge existence (such as
                collision detection) until the cost is queried.

        Raises:
            ValueError if `parent` or `child` is not the label of a
            vertex in the graph
        """
        if child in self.neighbors(parent):
            return 1.
        else:
            return float('inf')

    def is_connected(self):
        """Check if a graph is connected

        A graph is connected if there is a path between every pair of
        vertices.

        Returns:
            bool: True if the graph is connected

        Examples:
            >>> d1 = {0: {1, 2}, 1: {0}, 2: {0}}
            >>> HashGraph(d1).is_connected()
            True
            >>> d2 = {0: {1}, 1: {0}, 2: {}}
            >>> HashGraph(d2).is_connected()
            False
        """
        seen = set() # Vertices seen so far
        queue = set() # Vertices not yet seen

        # initial set includes the first vertex in iterator order
        queue.add(next(iter(self)))
        while queue:
            current = queue.pop()
            seen.add(current)
            neighbors = set(v for v in self.neighbors(current)
                            if self.cost(current, v) < float('inf'))
            queue |= neighbors - seen
        return len(seen) == len(self)

class HashDigraph(AbstractDigraph):
    """Simple implementation of an abstract digraph backed by a dict

    Args:
        successors (dict): maps labels to sequences of successor vertices
        data (dict): maps labels to vertex data
        cost (dict): maps edges (represented as 2-tuples of vertices) to
            their associated costs. Any edge not represented in
            successors is ignored, and any missing edge is assumed to
            have unit cost.

    Attributes:
        _successors (dict): maps labels to sets of succcessor vertices
        _predecessors (dict): maps labels to sets of predecessor vertices
        _data (dict): maps labels to vertex data
        _cost (dict): maps edges (represented as 2-tuples of vertices)
            to their associated costs
    """
    def __init__(self, successors, data=None, cost=None):
        self._successors = {}
        self._predecessors = {}
        self._data = {}
        self._cost = {}

        for label in successors:
            self._successors[label] = set(successors[label])
            self._predecessors[label] = set()
            if data is not None and label in data:
                self._data[label] = data[label]
            else:
                self._data[label] = None
            for successor in successors[label]:
                if cost is not None and (label, successor) in cost:
                    self._cost[(label, successor)] = cost[(label, successor)]

        for label in self:
            for successor in self.successors(label):
                self._predecessors[successor].add(label)

    def __contains__(self, vertex):
        """Check if vertex is the label of a vertex in the graph"""
        return vertex in self._successors

    def __iter__(self):
        """Iterate over the labels of vertices in the graph"""
        return iter(self._successors)

    def __len__(self):
        """Return the cardinality of the set of vertices"""
        return len(self._successors)

    def __getitem__(self, vertex):
        """Return the data associated with the vertex"""
        return self._data.get(vertex)

    def __setitem__(self, vertex, data):
        """Set the data associated with the vertex"""
        if vertex not in self:
            self._successors[vertex] = set()
            self._predecessors[vertex] = set()
        self._data[vertex] = data

    def __delitem__(self, vertex):
        """Remove a vertex from the graph"""
        for predecessor in self.predecessors(vertex):
            self._successors[predecessor].discard(vertex)
            self._cost.pop((predecessor, vertex), None)
        for successor in self.successors(vertex):
            self._predecessors[successor].discard(vertex)
            self._cost.pop((vertex, successor), None)

        self._data.pop(vertex, None)
        del self._predecessors[vertex]
        del self._successors[vertex]

    def predecessors(self, vertex):
        """Return the vertices with vertex as a child"""
        return self._predecessors[vertex]

    def successors(self, vertex):
        """Return the vertices with vertex as a parent"""
        return self._successors[vertex]

    def add_edge(self, parent, child, cost=None):
        """Add an edge from parent to child

        If either vertex is not in the graph, it will be added.
        """
        if parent not in self:
            self[parent] = None
        if child not in self:
            self[child] = None

        self._successors[parent].add(child)
        self._predecessors[child].add(parent)
        if cost is not None:
            self.set_cost(parent, child, cost)

    def remove_edge(self, parent, child):
        """Remove an edge from parent to child if it exists

        If either vertex is not in the graph, it will be added.
        """
        if parent in self and child in self:
            self._predecessors[child].discard(parent)
            self._successors[parent].discard(child)
            self._cost.pop((parent, child), None)

    def cost(self, parent, child):
        """Get the cost of an edge"""
        if child not in self._successors[parent]:
            return float('inf')
        else:
            return self._cost.get((parent, child), 1.)

    def set_cost(self, parent, child, cost):
        """Set the cost of an edge"""
        if child in self.successors(parent):
            self._cost[(parent, child)] = cost

class HashGraph(UndirectedGraph):
    """Simple implementation of an undirected graph backed by a dict

    Args:
        neighbors (dict): maps labels to sequences of neighbor vertices
        data (dict): maps labels to vertex data
        cost (dict): maps edges (represented as 2-tuples of vertices) to
            their associated costs. Any edge not represented in
            successors is ignored, and any missing edge is assumed to
            have unit cost.

    Attributes:
        _neighbors (dict): maps labels to sets of neighbor vertices
        _data (dict): maps labels to vertex data
        _cost (dict): maps edges (represented as 2-tuples of vertices)
            to their associated costs
    """
    def __init__(self, neighbors, data=None, cost=None):
        self._neighbors = {}
        self._data = {}
        self._cost = {}

        for vertex in neighbors:
            self._neighbors[vertex] = set(neighbors[vertex])
            if data is not None and vertex in data:
                self._data[vertex] = data[vertex]
            if cost is not None:
                for neighbor in neighbors[vertex]:
                    edge = (vertex, neighbor)
                    if edge in cost:
                        self._cost[edge] = cost[edge]

    def __contains__(self, vertex):
        """Check if vertex is the label of a vertex in the graph"""
        return vertex in self._neighbors

    def __iter__(self):
        """Iterate over the labels of vertices in the graph"""
        return iter(self._neighbors)

    def __len__(self):
        """Return the cardinality of the set of vertices"""
        return len(self._neighbors)

    def __getitem__(self, vertex):
        """Return the vertex data associated with the vertex"""
        return self._data.get(vertex) if vertex in self else None

    def __setitem__(self, vertex, data):
        """Set the vertex data associated with the vertex"""
        if vertex not in self:
            self._neighbors[vertex] = set()
        if data is not None:
            self._data[vertex] = data

    def __delitem__(self, vertex):
        """Remove a vertex from the graph"""
        for neighbor in self._neighbors[vertex]:
            self._neighbors[neighbor].discard(vertex)
            self._cost.pop((vertex, neighbor), None)

        self._data.pop(vertex)
        del self._neighbors[vertex]

    def neighbors(self, vertex):
        """Return the vertices adjacent to vertex"""
        return self._neighbors[vertex]

    def add_edge(self, parent, child, cost=None):
        """Add an edge from parent to child

        If either vertex is not in the graph, it will be added.
        """
        if parent not in self:
            self[parent] = None
        if child not in self:
            self[child] = None

        self._neighbors[parent].add(child)
        self._neighbors[child].add(parent)
        if cost is not None:
            self.set_cost(parent, child, cost)

    def remove_edge(self, parent, child):
        """Remove an edge from parent to child if it exists

        If either vertex is not in the graph, it will be added.
        """
        if parent in self and child in self:
            self._neighbors[parent].discard(child)
            self._neighbors[child].discard(parent)
            self._cost.pop((parent, child), None)

    def cost(self, parent, child):
        """Get the cost of an edge"""
        if child not in self._neighbors[parent]:
            return float('inf')
        elif parent < child:
            return self._cost.get((parent, child), 1)
        else:
            return self._cost.get((child, parent), 1)

    def set_cost(self, parent, child, cost):
        """Set the cost of an edge"""
        if child in self._neighbors[parent]:
            if parent < child:
                self._cost[(parent, child)] = cost
            else:
                self._cost[(child, parent)] = cost
