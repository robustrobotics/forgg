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

class FactoredRandomGeometricGraph(UndirectedGraph):
    def __init__(self, geometry, dynamics, robot, count):
        """
        Args:
            geometry (ManyShapeGeometry): used to perform collision detection
            robot (str): the identifier of the object to consider the
                robot for planning purposes
            dynamics: used to determine how each object can move
            count: number of poses of each object to sample
        """
        assert robot in geometry.bodies
        self.geometry = geometry
        self.dynamics = dynamics
        self.component_sets = {
            name: numpy.array([geometry.sample_object_configuration(name)
                               for _ in xrange(count)])
            for name in geometry.bodies}

        # Note the actual set of vertices is more or less the power set
        # of the component sets. It's actually a bit more complicated:
        # if S is the power set of the N configurations of each of the
        # K non-robot objects, the set of vertices is the cartesian
        # product of S and the union of the set of grasping poses for
        # each object pose for each object. That's (K+1) N Ng * N^K
        # objects represented in NK+(K+1)*N*Ng space

        dim = 3
        d_inv = 1. / float(dim)
        zeta_d = math.pi**(dim/2) / math.gamma(dim/2 + 1)
        self.epsilon = .5 #TODO: the amount to approximate search by
        # (>0; larger is faster but less accurate)
        self.eta = 2 # TODO: the amount to inflate the search radius by
        # (>1; larger is slower but more accurate)
        self.mu_free = geometry.mu_free # Lebesgue measure of free space
        self.search_radius = 2 * self.eta * (
            d_inv * (self.mu_free / zeta_d) * (math.log(count) / float(count))
            )**d_inv

        # There's some bookkeeping to figure out here. We need to index
        # the set of robot configurations such that they keep track of
        # if they are 'free' configurations, or tied to one of the other
        # objects. One way to do this would be to maintain a list L, so
        # that if configuration i was added due to configuration j of
        # object k, L[i] = (j, k). Then, the neighbors of any vertex
        # can be computed using by k-d tree on the set of all
        # configurations, and using the bookkeeping list L to select
        # only those vertices for which
        #   L[i][0]=vertex[L[i][1]] or L[i][1]==robot
        # That is, we keep only the poses that are computed due to
        # the current configuration

        # This is part of a kludgy hack to make it possible to look up
        # free robot poses in the modified component_sets
        self.free_robot_poses = self.component_sets[robot]
        # TODO: possible bug. Is free_robot_poses a copy of a reference
        # to the same data structure as component_sets[robot]?
        self.free_robot_pose_index = {}
        robot_poses = []
        bookkeeping = []
        for name, configurations in self.component_sets.iteritems():
            for index, configuration in enumerate(configurations):
                if name == robot:
                    robot_poses.append(configuration)
                    bookkeeping.append((len(robot_poses)-1, robot))
                    self.free_robot_pose_index[index] = len(robot_poses) - 1
                else:
                    for pose in dynamics.sample_contact_pose(
                            geometry.bodies[name], configuration):
                        robot_poses.append(pose)
                        bookkeeping.append((index, name))

        self.robot = robot
        self.component_sets[robot] = numpy.array(robot_poses)
        search = cKDTree(self.component_sets[robot] * geometry.scale)
        self.nearby_configurations = search.query_ball_tree(
            search, (1 + self.epsilon) * self.search_radius, eps=self.epsilon)
        self.bookkeeping = bookkeeping

    def __contains__(self, vertex):
        return all(0 <= i <= len(self.component_sets[name])
                   for name, i in vertex.iteritems())

    def __len__(self):
        return numpy.prod(len(s) for name, s in self.component_sets.iteritems())

    def __iter__(self):
        sets = {name: xrange(len(v))
                for name, v in self.component_sets.iteritems()}
        for component in itertools.product(*sets.itervalues()):
            yield metis.hashdict.hashdict(
                zip(self.component_sets.iterkeys(), component))

    def __getitem__(self, vertex):
        return {name: self.component_sets[name][i]
                for name, i in vertex.iteritems()}

    def adjacent(self, vertex1, vertex2):
        """Generate neighbors of vertex

        Two vertices are adjacent iff
        - The distance between their robot poses is less than the search
          radius and greater than zero
        And one of the following holds:
        - They share the same bookkeeping object: the same object
          generated both robot poses, so this is motion with an object
        - the bookkeeping pose for each vertex is the same as the pose
          of the bookkeeping object in the opposite vertex

        what does the second one mean? Imagine two vertices, v1 and v2,
        with bookkeeping objects and poses (o1, p1) and (o2, p2)
        respectively. If o1==o2, this vertex pair representes moving o1
        from pose p1 to pose p2. If neither o1 nor o2 is 'robot' and
        v1[o2]=p2 and v2[o1]=p1, this represents moving the robot from
        contact with o1 (at pose p1) to contact with o2 (at pose p2). If
        o1=='robot' and v1[o2]=p2, this represents picking up object o2
        (at pose p2). If o2=='robot' and v2[o1]=p1, this represents
        leaving object o1 behind at pose p1.
        """
        distance = 0 #TODO
        if distance > self.search_radius:
            return False
        else:
            p1, o1 = self.bookkeeping[vertex1[self.robot]]
            p2, o2 = self.bookkeeping[vertex2[self.robot]]
            return ((o1 == o2)
                    or (o1 == self.robot and vertex1[o2] == p2)
                    or (o2 == self.robot and vertex2[o1] == p1)
                    or (vertex1[o2] == p2 and vertex2[o1] == p1))

    def nearest(self, configuration):
        """Return the nearest vertex to configuration
        """
        # TODO this does not check if the configuration is free. I think
        # the easiest way to do that is to form the k nearest
        # configurations and then return the closest one which is
        # collision free
        vertex = {}
        for name, samples in self.component_sets.iteritems():
            if name == self.robot:
                search = cKDTree(self.free_robot_poses * self.geometry.scale)
                _, index = search.query(configuration[name])
                nearest = self.free_robot_pose_index[index]
            else:
                search = cKDTree(samples * self.geometry.scale)
                _, nearest = search.query(configuration[name])
            vertex[name] = nearest
        return metis.hashdict.hashdict(vertex)

    def neighbors(self, vertex):
        """Generate neighbors of vertex

        We can construct the adjacent vertices by considering all robot
        poses within search_radius and analyzing their bookkeeping
        objects and poses.
        """
        if not self.geometry.configuration_is_free(self[vertex]):
            raise StopIteration
        robot_pose = vertex[self.robot]
        neighbor = dict(vertex)
        source_pose, source_obj = self.bookkeeping[robot_pose]
        for neighbor_pose in self.nearby_configurations[robot_pose]:
            if neighbor_pose == robot_pose:
                continue

            dest_pose, dest_obj = self.bookkeeping[neighbor_pose]

            neighbor = dict(vertex)
            neighbor[self.robot] = neighbor_pose
            neighbor[dest_obj] = dest_pose

            if ((source_obj == dest_obj)
                    or (source_obj == self.robot and
                        vertex[dest_obj] == dest_pose)
                    or (dest_obj == self.robot and
                        neighbor[source_obj] == source_pose)
                    or (vertex[dest_obj] == dest_pose and
                        neighbor[source_obj] == source_pose)):
                if self.geometry.configuration_is_free(self[neighbor]):
                    yield metis.hashdict.hashdict(neighbor)

    def cost(self, parent, child):
        parent_configuration = {
            name: self.component_sets[name][i]
            for name, i in parent.iteritems()}
        child_configuration = {
            name: self.component_sets[name][i]
            for name, i in child.iteritems() if parent[name] != child[name]}

        # TODO: this does not check for collision between objects. I
        # *think* what we want is for any object that doesn't move to be
        # treated as part of the background, and any object that does
        # move to be ignored for collision detection. Not sure how to
        # accomplish that.
        if self.geometry.path_is_free(parent_configuration,
                                      child_configuration):
            return self.dynamics.cost(parent_configuration, child_configuration)
        else:
            return float('inf')

