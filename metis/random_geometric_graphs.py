"""Implements undirected random geometric graphs

A random geometric graph is a graph whose vertices correspond to
randomly sampled points on a manifold and whose edges are defined by
geometric functions. This principally includes r-disc graphs, where an
edge is defined between any pair of vertices separated by a geodesic
distance less than r.

The classes in this module take objects parameterizing dynamics and
geometry, and do the bookkeeping necessary to maintain a random
geometric graph.
"""

import math
import itertools

import numpy
from scipy.spatial import Delaunay, cKDTree, distance

import metis
from metis.abstract_graphs import UndirectedGraph

class RandomPlanarGraph(UndirectedGraph):
    """Random planar graph in R^2

    This is mostly useful for debugging and demonstration; because it is
    planar, it is easier to parse visually than an RGG. The construction
    employs uses a Delauanay triangulation to select edges, optionally
    removing any edges which intersect some geometry.

    Vertex labels are just indices of an array of points, so any integer
    between 0 and `count-1`.

    Args:
        geometry (metis.geometry.Geometry): object describing problem geometry
        count (int): number of vertices in the graph

    Attributes:
        geometry (metis.geometry.Geometry): object describing problem geometry
        configurations (numpy.array): a (count x 2) array of samples
            corresponding to vertices. Each row is a sample from the
            2D configuration space defined by the geometry.
        delaunay: the Delaunay triangulation of the vertices in the graph.
    """
    def __init__(self, geometry, count):
        super(RandomPlanarGraph, self).__init__()
        self.geometry = geometry
        configurations = numpy.array([
            geometry.sample_configuration() for _ in xrange(5 * count)])
        self.configurations = metis.geometry.sparsify_point_set(
            configurations, count)
        self.delaunay = Delaunay(self.configurations, incremental=True)

    def __contains__(self, vertex):
        return 0 <= vertex < len(self.configurations)

    def __repr__(self):
        return "RandomPlanarGraph(configurations={})".format(
            repr(self.configurations))

    def __str__(self):
        return "RandomPlanarGraph(count={})".format(len(self.configurations))

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        return iter(xrange(len(self.configurations)))

    def __getitem__(self, vertex):
        return self.configurations[vertex]

    def neighbors(self, vertex):
        """Generate neighbor vertices in Delaunay triangulation

        The Delaunay graph is guaranteed to be planar, which makes for
        easy viewing. Neighbor lookup is computationally inexpensive due
        to precomputation.

        Note:
            Does not perform collision detection; some edges returned
            here may not be collision free. Collision detection is
            performed when the cost of an edge is evaluated.

        Args:
            vertex (int): index of a vertex in the graph

        Returns:
            generator(int): generates all neighbors of vertex
        """
        indices, indptr = self.delaunay.vertex_neighbor_vertices
        for neighbor in indptr[indices[vertex]:indices[vertex+1]]:
            yield neighbor

    def cost(self, parent, child):
        """Compute edge cost for vertex pair

        Perform collision detection and evaluate the cost of an edge. If
        the edge is collision free, its cost is the distance between
        edges; if the edge is in collision, its cost is infinite.

        Args:
            parent (int): index of first vertex of edge
            child (int): index of second vertex of edge

        Returns:
            float: infinite if edge is in collision, otherwise the
                length of the edge
        """
        if self.geometry.path_is_free(self.configurations[parent],
                                      self.configurations[child]):
            return distance.euclidean(self.configurations[parent],
                                      self.configurations[child])
        else:
            return float('inf')

class EuclideanRandomGraph(UndirectedGraph):
    """Random disc graph in R^n

    A random disc graph has vertices sampled from an n dimensional
    manifold, and an edge between all vertices with distance less than a
    fixed constant. We take that constant to be the critical distance
    derived in Karaman and Frazzoli (2011), which is minimal distance
    required to ensure the resulting graph includes an optimal path.

    Vertex labels are just indices of an array of points, so any integer
    between 0 and `count-1`.

    Args:
        geometry (metis.geometry.Geometry): object describing problem geometry
        count (int): number of vertices in the graph
        epsilon (float): positive scalar multiplier by which to
            approximate nearest neighbor search (larger is faster but
            less accurate)
        eta (float): positive scalar multiplier by which to inflate the
            distance threshhold defining the graph (larger is slower but
            more accurate; graph is asymptotically optimal iff eta>=1)
        configurations (array-like): if supplied, include these
            configurations as vertices in the graph, in addition to
            `count` random configurations.

    Attributes:
        geometry (metis.geometry.Geometry): object describing problem geometry
        configurations (numpy.array): array of samples corresponding to
            vertices. Each row is a sample from the configuration space
            defined by geometry.
        epsilon (float): positive scalar multiplier by which to
            approximate nearest neighbor search (larger is faster but
            less accurate)
        eta (float): positive scalar multiplier by which to inflate the
            distance threshhold defining the graph (larger is slower but
            more accurate; graph is asymptotically optimal iff eta>=1)
        mu_free (float); positive scalar approximating the Lebesgue
            measure of free space in the configuration
        search_radius (float): positive scalar defining maximum distance
            between vertices for which an edge may exist
        nearby_configurations (list(list)): list of lists of neighboring
            vertices for each vertex in the graph. Precomputed for speed
    """
    def __init__(self, geometry, count, epsilon=0., eta=1., configurations=None):
        super(EuclideanRandomGraph, self).__init__()
        self.geometry = geometry

        if configurations is None:
            configurations = []
        configurations += [geometry.sample_configuration()
                            for _ in xrange(count)]
        self.configurations = numpy.array(configurations)

        dim = self.configurations.shape[1]
        d_inv = 1. / float(dim)
        zeta_d = math.pi**(dim/2) / math.gamma(dim/2 + 1)
        self.epsilon = epsilon
        self.eta = eta
        self.mu_free = geometry.mu_free # Lebesgue measure of free space
        self.search_radius = 2 * self.eta * (
            d_inv * (self.mu_free / zeta_d) * (math.log(count) / float(count))
            )**d_inv

        search = cKDTree(self.configurations * geometry.scale)
        self.nearby_configurations = search.query_ball_tree(
            search, (1 + self.epsilon) * self.search_radius, eps=self.epsilon)

    def __contains__(self, vertex):
        return 0 <= vertex < len(self.configurations)

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        return iter(xrange(len(self.configurations)))

    def __getitem__(self, vertex):
        return self.configurations[vertex]

    def neighbors(self, vertex):
        """Generate all vertices less than threshhold from vertex

        This is computationally inexpensive due to precomputation.
        Note:
            Does not perform collision detection; some edges returned
            here may not be collision free. Collision detection is
            performed when the cost of an edge is evaluated.

        Args:
            vertex (int): index of a vertex in the graph

        Returns:
            generator(int): generates all neighbors of vertex
        """
        return (n for n in self.nearby_configurations[vertex] if n != vertex)

    def cost(self, parent, child):
        """Compute edge cost for vertex pair

        Perform collision detection and evaluate the cost of an edge. If
        the edge is collision free, its cost is the distance between
        edges; if the edge is in collision, its cost is infinite.

        Args:
            parent (int): index of first vertex of edge
            child (int): index of second vertex of edge

        Returns:
            float: infinite if edge is in collision, otherwise the
                length of the edge
        """
        if self.geometry.path_is_free(self.configurations[parent],
                                      self.configurations[child]):
            return distance.euclidean(self.configurations[parent],
                                      self.configurations[child])
        else:
            return float('inf')

class SE2RandomGraph(UndirectedGraph):
    """Random disc graph in SE(2)

    A random disc graph has vertices sampled from an n dimensional
    manifold, and an edge between all vertices with distance less than a
    fixed constant. We take that constant to be the critical distance
    derived in Karaman and Frazzoli (2011), which is minimal distance
    required to ensure the resulting graph includes an optimal path.

    Vertex labels are just indices of an array of points, so any integer
    between 0 and `count-1`.

    Args:
        geometry (metis.geometry.Geometry): object describing problem geometry
        count (int): number of vertices in the graph
        epsilon (float): positive scalar multiplier by which to
            approximate nearest neighbor search (larger is faster but
            less accurate)
        eta (float): positive scalar multiplier by which to inflate the
            distance threshhold defining the graph (larger is slower but
            more accurate; graph is asymptotically optimal iff eta>=1)
        configurations (array-like): if supplied, include these
            configurations as vertices in the graph, in addition to
            `count` random configurations.

    Attributes:
        geometry (metis.geometry.Geometry): object describing problem geometry
        configurations (numpy.array): array of samples corresponding to
            vertices. Each row is a sample from the configuration space
            defined by geometry.
        epsilon (float): positive scalar multiplier by which to
            approximate nearest neighbor search (larger is faster but
            less accurate)
        eta (float): positive scalar multiplier by which to inflate the
            distance threshhold defining the graph (larger is slower but
            more accurate; graph is asymptotically optimal iff eta>=1)
        mu_free (float); positive scalar approximating the Lebesgue
            measure of free space in the configuration
        search_radius (float): positive scalar defining maximum distance
            between vertices for which an edge may exist
        nearby_configurations (list(list)): list of lists of neighboring
            vertices for each vertex in the graph. Precomputed for speed
    """
    def __init__(self, geometry, count, epsilon=0., eta=1.,
                 configurations=None):
        super(SE2RandomGraph, self).__init__()
        self.geometry = geometry
        self.count = count

        if configurations is None:
            configurations = []
        configurations.extend(geometry.sample_configuration()
                              for _ in xrange(count))
        self.configurations = numpy.array(configurations)

        dim = self.configurations.shape[1]
        d_inv = 1. / float(dim)
        zeta_d = math.pi**(dim/2) / math.gamma(dim/2 + 1)
        self.epsilon = epsilon
        self.eta = eta
        self.mu_free = geometry.mu_free # Lebesgue measure of free space
        self.search_radius = 2 * self.eta * (
            d_inv * (self.mu_free / zeta_d) * (math.log(count) / float(count))
            )**d_inv

        search = cKDTree(self.configurations * geometry.scale)
        self.nearby_configurations = search.query_ball_tree(
            search, (1 + self.epsilon) * self.search_radius, eps=self.epsilon)

    def __contains__(self, vertex):
        return 0 <= vertex < len(self.configurations)

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        return iter(xrange(len(self.configurations)))

    def __getitem__(self, vertex):
        return self.configurations[vertex]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['geometry']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.geometry = None

    def neighbors(self, vertex):
        """Generate all vertices less than threshhold from vertex

        This is computationally inexpensive due to precomputation.
        Note:
            Does not perform collision detection; some edges returned
            here may not be collision free. Collision detection is
            performed when the cost of an edge is evaluated.

        Args:
            vertex (int): index of a vertex in the graph

        Returns:
            generator(int): generates all neighbors of vertex
        """
        return (n for n in self.nearby_configurations[vertex] if n != vertex)

    def cost(self, parent, child):
        """Compute edge cost for vertex pair

        Perform collision detection and evaluate the cost of an edge. If
        the edge is collision free, its cost is the distance between
        edges; if the edge is in collision, its cost is infinite.

        Args:
            parent (int): index of first vertex of edge
            child (int): index of second vertex of edge

        Returns:
            float: infinite if edge is in collision, otherwise the
                length of the edge
        """

        parent = self.configurations[parent]
        child = self.configurations[child]
        if self.geometry.path_is_free(parent, child):
            scale = self.geometry.scale
            dtheta = math.fabs(parent[2] - child[2])
            dtheta = min(dtheta, 2*math.pi-dtheta)
            return math.sqrt((scale[0] * (parent[0]-child[0]))**2 +
                             (scale[1] * (parent[1]-child[1]))**2 +
                             (scale[2] * dtheta)**2)
        else:
            return float('inf')

