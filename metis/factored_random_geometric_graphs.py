"""Implements a factored random geometric graph

A factored random geometric graph is special representation of a random
geometric graph, for the case when the configuration space is the
Cartesian product of a numbor of component manifolds. The particular
case of interest is the manifold of configurations of K objects, which
is the Cartesian product of K copies SE(2) (in two dimensions) or SE(3)
in three dimensions.

The vertices can be represented by length-K tuples, where each element
is an index into a sequence of configurations in one of the component
manifolds. The precise form of edges required to ensure probabilistic
completeness and asymptotic optimality is an object of ongoing research.

The classes in this module take objects parameterizing dynamics and
geometry, and do the bookkeeping necessary to maintain a factored random
geometric graph.
"""
import math
import itertools

import numpy
from scipy.spatial import cKDTree

import metis
from metis.abstract_graphs import UndirectedGraph

class NoObjectContactBlacklist(object):
    def __init__(self, robot='robot'):
        super(NoObjectContactBlacklist, self).__init__()
        self.robot = robot

    def __contains__(self, pair):
        return pair[0] != self.robot and pair[1] is not None

def apply_transform(tfA, tfB):
    """Return the result of applying tfA to tfB

    Args:
        tfA (tuple): a transform, represented as an (x, y, theta) tuple
        tfB (tuple): a transform, represented as an (x, y, theta) tuple

    Returns:
        tuple: (x, y, theta) representing a transform with the same
            effect as first applying 'tfB' and then applying 'tfA'. If
            the transforms were represented as matrices, this would be
            tfA * tfB.
    """
    return (tfA[0] + numpy.cos(tfA[2]) * tfB[0] - numpy.sin(tfA[2]) * tfB[1],
            tfA[1] + numpy.sin(tfA[2]) * tfB[0] + numpy.cos(tfA[2]) * tfB[1],
            tfA[2] + tfB[2])

def apply_inverse_transform(tfA, tfB):
    """Return the result of applying tfA to the inverse of tfB

    Args:
        tfA (tuple): a transform, represented as an (x, y, theta) tuple
        tfB (tuple): a transform, represented as an (x, y, theta) tuple

    Returns:
        tuple: (x, y, theta) representing a transform with the same
            effect as first applying the inverse of 'tfB' and then
            applying 'tfA'. If the transforms were represented as
            matrices, this would be tfA / tfB.
    """
    dtf = tfA - tfB
    return (numpy.cos(tfB[2]) * dtf[0] + numpy.sin(tfB[2]) * dtf[1],
            -numpy.sin(tfB[2]) * dtf[0] + numpy.cos(tfB[2]) * dtf[1],
            dtf[2])

class Manifold(object):
    def __init__(self, samples):
        self.samples = numpy.array(samples)
        assert self.samples.ndim == 2, "Data must be a sequence of samples"

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(xrange(len(self.samples)))

    def __contains__(self, index):
        return 0 <= index < len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def nearest_samples(self, point, k):
        """Return then indices of the samples near the query pose"""
        pass

    def nearby_samples(self, point, r):
        """Return then indices of the samples near the query pose"""
        pass


    def neighbors(self, index):
        """Return the neighbors of the specified sample index"""
        pass


class SE2Manifold(Manifold):
    """Random disc graph on a manifold

    A random disc graph has vertices sampled from an n dimensional
    manifold, and an edge between all vertices with minimal geodesic
    distance less than a fixed constant. We take that constant to be the
    critical distance derived in Karaman and Frazzoli (2011), which is
    minimal distance required to ensure the resulting graph includes an
    optimal path.

    Vertex labels are just indices of an array of points, so any integer
    between 0 and `count-1`.

    Args:
        samples (array-like): if supplied, include these
            configurations as vertices in the graph, in addition to
            `count` random configurations.
        epsilon (float): positive scalar multiplier by which to
            approximate nearest neighbor search (larger is faster but
            less accurate)
        eta (float): positive scalar multiplier by which to inflate the
            distance threshhold defining the graph (larger is slower but
            more accurate; graph is asymptotically optimal iff eta>=1)

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
    def __init__(self, samples, scale=1., epsilon=0., search_radius=None, eta=1.):
        super(SE2Manifold, self).__init__(samples)
        assert self.samples.shape[1] == 3, \
            "Data must be 'x, y, theta' sequences"

        self.scale = scale
        self.epsilon = epsilon
        self.eta = eta

        if search_radius is None:
            count, dim = self.samples.shape
            d_inv = 1. / float(dim)
            zeta_d = math.pi**(dim/2) / math.gamma(dim/2 + 1)

            mu_free = (numpy.prod(numpy.ptp(self.samples[:,:-1], axis=0))
                       * 2 * numpy.pi * scale)
            self.search_radius = 2 * self.eta * (
                d_inv * (mu_free / zeta_d) * (math.log(count) / float(count))
                )**d_inv
        else:
            self.search_radius = search_radius

        self._search = cKDTree(numpy.concatenate(
            (self.samples[:, :-1],
             self.scale * numpy.cos(self.samples[:, -1:]),
             self.scale * numpy.sin(self.samples[:, -1:])), axis=1))
        self._neighbors = self._search.query_ball_tree(
            self._search, (1 + epsilon) * self.search_radius, eps=epsilon)

    def nearest_samples(self, point, k):
        """Return the indices of the k samples nearest the query point

        Args:
            point: query location, represented as an (x, y, theta) tuple

        Returns:
            list: list of tuples (d, i), where d is the distance between
                sample i and the query point
        """
        query = (point[0], point[1],
                 self.scale * numpy.cos(point[2]),
                 self.scale * numpy.sin(point[2]))
        return zip(*self._search.query(query, k))

    def nearby_samples(self, point, r):
        """Return the indices of the samples near the query point

        Args:
            point: query location, represented as an (x, y, theta) tuple
            r: search radius

        Returns:
            list: list of indices of samples within distance r of the
                query point
        """
        query = (point[0], point[1],
                 self.scale * numpy.cos(point[2]),
                 self.scale * numpy.sin(point[2]))
        return self._search.query_ball_point(
            query, (1+self.epsilon)*r, eps=self.epsilon)

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
        return (n for n in self._neighbors[vertex] if n != vertex)

    def distance(self, pose1, pose2):
        point1 = numpy.array((pose1[0], pose1[1],
                              self.scale * numpy.cos(pose1[2]),
                              self.scale * numpy.sin(pose1[2])))
        point2 = numpy.array((pose2[0], pose2[1],
                              self.scale * numpy.cos(pose2[2]),
                              self.scale * numpy.sin(pose2[2])))
        return numpy.sqrt(numpy.sum((point1-point2)**2))

class FactoredRandomGeometricGraph(UndirectedGraph):
    """Factored random geometric graph for manipulation

    The vertices of this graph are hashable dict-like structures mapping
    object names to tuples of (parent name, transform index), where
    parent_name is the name of the object relative to which the
    transform of the object name is computed. A vertex can be unpacked into
    configurations using the notation graph[vertex]. The graph is
    immutable after construction; vertex data cannot be changed or
    deleted.

    Args:
        geometry (ManyShapeGeometry): object describing problem geometry
        dynamics: used to determine how each object can move
        robot (str): the identifier of the object to consider the
            robot for planning purposes
        geometry (metis.geometry.ManyShapeGeometry): object describing
            problem geometry
        counts (dict): maps manifolds, specified as (object, object)
            2-tuples, to the number of random samples to draw from that
            manifold
        default_count (int): number of poses of each object to sample.
            This value is used for any manifold not specified in counts,
            and is overridden by any manifold specified in counts.
        configurations (dict): if supplied, include these
            configurations in the component set for the supplied
            objects, in addition to `count` random configurations.
        epsilon (float): positive scalar multiplier by which to
            approximate nearest neighbor search (larger is faster but
            less accurate)
        eta (float): positive scalar multiplier by which to inflate the
            distance threshhold defining the graph (larger is slower but
            more accurate; graph is asymptotically optimal iff eta>=1)

    Attributes:
        geometry (ManyShapeGeometry): object describing problem geometry
        dynamics: used to determine how each object can move
        robot (str): the identifier of the object to consider the
            robot for planning purposes
        manifolds (dict): maps manifold names to structures defining the
            manifolds themselves
        epsilon (float): positive scalar multiplier by which to
            approximate nearest neighbor search (larger is faster but
            less accurate)
        eta (float): positive scalar multiplier by which to inflate the
            distance threshhold defining the graph (larger is slower but
            more accurate; graph is asymptotically optimal iff eta>=1)
    """
    def __init__(self, geometry, dynamics, counts=None, default_count=0,
                 configurations=None, blacklist=(), robot='robot',
                 epsilon=0., eta=1.):
        # pylint: disable=too-many-arguments
        assert robot in geometry.bodies

        # Rebind default mutable arguments
        if counts is None:
            counts = {}
        if configurations is None:
            configurations = {}

        # Declare members
        self.geometry = geometry
        self.dynamics = dynamics
        self.names = [name for name in geometry.bodies]
        self.samples = {}
        self.robot = robot
        self.epsilon = epsilon
        self.eta = eta

        for name in self.names:
            transforms = configurations.get((name, None), [])
            count = counts.get((name, None), default_count)
            transforms += [geometry.sample_object_configuration(name)
                           for _ in xrange(count)]
            self.samples[(name, None)] = SE2Manifold(transforms, epsilon=epsilon, eta=eta)

            # Sample poses relative to other objects
            for other in self.names:
                if name != other and (name, other) not in blacklist:
                    count = counts.get((name, other), default_count)
                    poses = list(dynamics.sample_from_manifold((name, other), count))
                    self.samples[(name, other)] = SE2Manifold(
                        [pose for pose in poses])
                else:
                    self.samples[(name, other)] = Manifold([[]])

    @staticmethod
    def is_acyclic(vertex):
        """Determine if a directed chain graph is acyclic

        An acyclic chain is one name for a directed graph where each
        vertex is the source of at most one outward edge, and there are
        no cycles of directed edges. So for vertices (A, B, C), A=>B=>C
        would be an acyclic chain, while A=>B=>C=>A would not.

        Args:
            vertex (dict): maps vertex labels to a tuple (parent, None),
                describing the unique edge beginning at the vertex. A
                missing edge is denoted as an edge back to the vertex,
                so `{name: (name, None)}` describes a vertex with label
                `name` and no outgoing edge. The choice to represent an
                edge as a tuple with second value None was made for
                uniformity with the vertex representation, which uses
                the second argument to specify an index.

        Yields:
            bool: true if the graph has no cycles
        """
        todo = set(vertex.iterkeys())
        while todo:
            name = todo.pop()
            chain = set()
            while True:
                parent, _ = vertex[name]
                chain.add(name)
                if parent is None:
                    # We've found a root node.
                    break
                elif parent in chain:
                    return False
                elif parent in todo:
                    # If we haven't already confirmed this node is not
                    # part of a cycle, continue searching this chain
                    name = parent
                else:
                    # If we haven't already confirmed this node is not
                    # part of a cycle, continue searching this chain
                    break
            todo -= chain
        return True

    @staticmethod
    def acyclic_chains(names):
        """Generate the acyclic chains with labels drawn from names

        An acyclic chain is one name for a directed graph where each
        vertex is the source of at most one outward edge, and there are
        no cycles of directed edges. So for vertices (A, B, C), A=>B=>C
        would be an acyclic chain, while A=>B=>C=>A would not.

        Note there are very many acyclic chains: if there are $n$ names,
        there are at least $n!$ acyclic chains, and at most $n^n$. This
        function computes the acyclic chains by sequentially generating
        all directed graphs with maximal outdegree 1, then rejecting
        those graphs with cycles. The running time of this function is
        thus `O(len(names)^len(names))`, also known as really really
        slow.

        Args:
            names (list): list of vertex labels

        Yields:
            dict: maps each name to a tuple (parent, None), where parent
                is the sink of its unique edge. A missing edge is
                denoted as an edge back to the vertex, so `{name: (name,
                None)}` describes a vertex with label `name` and no
                outgoing edge. The choice to represent an edge as a
                tuple with second value None was made for uniformity
                with the vertex representation, which uses the second
                argument to specify an index..
        """
        for combination in  itertools.product(*(
                [(a, b if b != a else None) for b in names]
                for a in names)):
            vertex = {name: (parent, None) for name, parent in combination}
            if FactoredRandomGeometricGraph.is_acyclic(vertex):
                yield vertex

    def __contains__(self, vertex):
        """Check if the graph contains a vertex

        The graph contains a vertex if the keys are all the object
        names, the values are tuples (parent, index), the index is a key
        for the (object, parent) sample set, and the directed graph
        formed with edges between each object and its parent is acyclic.
        """
        for name in self.names:
            if name not in vertex:
                return False
            (parent, index) = vertex[name]
            if (name, parent) not in self.samples:
                return False
            if index not in self.samples[(name, parent)]:
                return False
        if not self.is_acyclic(vertex):
            return False
        return True

    def __len__(self):
        """Check if the graph contains a vertex

        The graph contains a vertex if the keys are all the object
        names, the values are tuples (parent, index), the index is a key
        for the (object, parent) sample set, and the directed graph
        formed with edges between each object and its parent is acyclic.
        """
        total = 0
        for vertex in self.acyclic_chains(self.names):
            prod = 1
            for name, (parent, _) in vertex.iteritems():
                prod *= len(self.samples[(name, parent)])
            total += prod
        return total

    def __iter__(self):
        """Iterate over all vertices in the graph

        Vertices are generated by sequentially generating acyclic
        chains, then generating all combinations of vertex indices for
        those acyclic chains. Note that the size of the graph is
        superexponential in the number of objects, so it can take a very
        long time to iterate over all vertices, even for small,
        low-resolution graphs.
        """
        for vertex in self.acyclic_chains(self.names):
            for indices in itertools.product(*(
                    iter(self.samples[(name, parent)])
                    for name, (parent, _) in vertex.iteritems())):
                yield {name: (parent, index)
                       for (name, (parent, _)), index
                       in zip(vertex.iteritems(), indices)}

    def get_pose_of(self, vertex, name):
        """Recursively compute the pose of an object at a vertex"""
        parent, index = vertex[name]
        transform = self.samples[(name, parent)][index]
        if parent is None:
            return transform
        else:
            return apply_transform(self.get_pose_of(vertex, parent), transform)

    def __getitem__(self, vertex):
        """Compute the configuration associated with a vertex

        Because vertices represent the pose of each object relative to
        the pose of another object, computing the absolute pose of each
        object requires first computing a topological ordering on the
        chain, then propagating an absolute pose down the ordering. This
        could be done by maintaining the vertex structure in a way that
        makes it easy to identify root poses; however, that would
        increase the size and complexity of the vertex representation.
        Instead, we compute the topological ordering implicity by
        iterating over the objects in an arbitrary order and keeping a
        'to-do list' of objects discovered whose parents had not yet
        been grounded.

        Args:
            vertex (dict)

        Returns:
            dict: maps object names to (x, y, theta) tuples
        """
        todo = {}
        result = {}
        for name in vertex:
            parent, index = vertex[name]
            if parent is None:
                result[name] = self.samples[(name, parent)][index]
            elif parent in result:
                transform = self.samples[(name, parent)][index]
                result[name] = apply_transform(result[parent], transform)
            else:
                todo.setdefault(parent, set()).add(name)

            if name in todo:
                for child in todo[name]:
                    expected_name, index = vertex[child]
                    assert expected_name == name
                    transform = self.samples[(child, name)][index]
                    result[child] = apply_transform(result[name], transform)
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['geometry']
        del state['dynamics']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.geometry = None
        self.dynamics = None

    def nearest(self, configuration):
        """Return the closest collision-free vertex to configuration

        Distance is the sum of the Euclidean distances between the pose
        of each object in the configuration and in the vertex. This is
        computed in a reasonably efficient way using dynamic programming.

        Args:
            configuration (dict): maps object names to poses

        Returns:
            dict: a dictionary representing the closest collision-free
                vertex to configuration
        """
        def expand_queue(name, k):
            """Return a sorted list of (distance, index) tuples"""
            return sorted(self.samples[(name, None)].nearest_samples(
                configuration[name], k))

        initial_k = 8
        sample_queues = {}
        for name in configuration:
            sample_queues[name] = expand_queue(name, initial_k)

        distance = lambda v: sum(sample_queues[name][v[name]][0]
                                 for name in configuration)
        start = metis.hashdict.hashdict({name: 0 for name in configuration})
        queue = metis.queue.PriorityQueue([(start, distance(start))])

        searched = 0
        while searched < len(self):
            current = queue.pop()
            current_vertex = metis.hashdict.hashdict({
                name: (None, sample_queues[name][index][1])
                for name, index in current.iteritems()})
            searched += 1
            if self.geometry.configuration_is_free(
                    self[current_vertex], skip_static=True):
                return current_vertex
            else:
                for name in current:
                    child = current + {name: current[name] + 1}
                    k = len(sample_queues[name])
                    if child[name] == k:
                        k = min(len(self.samples[(name, None)]), k * 2)
                        sample_queues[name] = expand_queue(name, k)
                    if child not in queue:
                        queue[child] = distance(child)

    def neighbors(self, vertex):
        """Generate neighbors of vertex

        We can construct the adjacent vertices by considering all
        reachable acyclic chains, then identifying within each chain all
        configurations within a distance of search_radius from the
        current vertex.

        Args:
            vertex (dict): maps object names to tuples (parent, i), with
                i an index in the sample set (name, parent)

        Yields:
            hashdict: neighboring vertices on the factored graph
        """
        # Check if the configuration encoded in the vertex dict is free
        if not self.geometry.configuration_is_free(
                self[vertex], skip_static=True):
            raise StopIteration

        result = metis.hashdict.hashdict(vertex)

        # TODO: Many things in this function should be in the dynamics.
        # Instead they are hard-coded for convenience

        # Identify the parent and the root of the current vertex (the
        # root is the ancestor of 'robot' whose parent is None)
        parent, _ = vertex[self.robot]
        root_parent = root = self.robot
        while root_parent is not None:
            root = root_parent
            root_parent, root_index = vertex[root]

        # Generate neighbors on the current manifold [D]
        for sample in self.samples[(root, None)].neighbors(root_index):
            neighbor = result + {root: (None, sample)}
            if self.geometry.configuration_is_free(
                    self[neighbor], skip_static=True):
                yield neighbor

        # Generate neighbors on adjacent manifolds.  The adjacent
        # acyclic chains are hard-coded as the chains for which only the
        # robot changes its parent relative to the chain described in
        # vertex.

        # We always move through free space to change parents, so use
        # the search radius for free space
        radius = self.samples[(self.robot, None)].search_radius
        if self.robot != root:
            # The robot is not a root pose: it is currently in contact
            # with something. Look for poses not in contact with
            # anything
            pose = self.get_pose_of(vertex, self.robot)
            manifold = self.samples[(self.robot, None)]
            for sample in manifold.nearby_samples(pose, radius):
                neighbor = result + {self.robot: (None, sample)}
                if self.geometry.configuration_is_free(
                        self[neighbor], skip_static=True):
                    yield neighbor

        for new_parent in self.names:
            # Consider changing parent to non-root poses
            if new_parent == self.robot or new_parent == parent:
                continue
            else:
                # Now we find nearby poses in this cycle. Translate the
                # current pose to a pose relative to the new parent
                pose = self.get_pose_of(vertex, self.robot)
                parent_pose = self.get_pose_of(vertex, new_parent)
                relative_pose = apply_inverse_transform(pose, parent_pose)

                # Search for neighbors near the absolute pose
                manifold = self.samples[(self.robot, new_parent)]
                for sample in manifold.nearby_samples(relative_pose, radius):
                    neighbor = result + {self.robot: (new_parent, sample)}
                    if self.geometry.configuration_is_free(
                            self[neighbor], skip_static=True):
                        yield neighbor

    def cost(self, parent, child):
        parent_configuration = self[parent]
        child_configuration = {self.robot: self.get_pose_of(child, self.robot)}
        holding = child[self.robot][0]
        if holding is not None:
            child_configuration[holding] = self.get_pose_of(child, holding)
        was_holding = parent[self.robot][0]
        if was_holding is not None:
            child_configuration[was_holding] = self.get_pose_of(child, was_holding)
        # child_configuration = {
        #     name: self.get_pose_of(child, name)
        #     for name in child if parent[name] != child[name]}

        # TODO: this does not check for collision between objects. I
        # *think* what we want is for any object that doesn't move to be
        # treated as part of the background, and any object that does
        # move to be ignored for collision detection. Not sure how to
        # accomplish that.
        if self.geometry.path_is_free(parent_configuration,
                                      child_configuration,
                                      skip_configuration=True):
            return self.dynamics.cost(parent_configuration, child_configuration)
        else:
            return float('inf')


