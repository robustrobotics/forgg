"""Encapsulate problem dynamics

I haven't quite resolved the correct API for dynamics; that will
probably happen as I develop the theory of factored Bellman trees. For
the time being, I'm just writing functions as I need them, and I'll try
to come up with a coherent logic later.
"""
import math

import numpy

import metis.geometry
import metis.iterators

class MagneticDynamics(object):
    """Simple manipulation dynamics

    In this dynamics model, an agent can apply unlimited force and
    torque to an object if it is in a grasping pose for that object:
    they can effectively be treated as one large object. The agent can
    also release an object, like turning off an electromagnet.
    """
    def __init__(self, bodies, agent='robot'):
        self.bodies = bodies
        self.agent = agent
        self.shapes = {
            name: metis.geometry.shapely_from_box2d_body(body, (0, 0, 0))
            for name, body in bodies.iteritems()}

    def sample_from_manifold(self, manifold, count, epsilon=1e-2):
        """Return count poses from the manifold

        Ignore 'count' for now

        Args:
            epsilon (float): shrink both objects by this much to prevent
                collision detection bugs
        """
        if manifold[0] == self.agent:
            assert manifold[1] is not None
            agent = self.shapes[manifold[0]]
            entity = self.shapes[manifold[1]]
            edges = metis.iterators.sequential_pairs(entity.boundary.coords)
            for (first, second) in edges:
                first = numpy.array(first)
                second = numpy.array(second)
                midpoint = (first + second) / 2
                delta = second - first
                heading = numpy.array((-delta[1], delta[0]))

                # Assume the first vertex is the front
                first_coord = numpy.array(agent.boundary.coords)[0]

                cross = numpy.cross(first_coord, heading)
                dot = numpy.dot(first_coord, heading)
                scale = numpy.linalg.norm(first_coord) / numpy.linalg.norm(heading)

                theta = math.atan2(cross, dot)
                # Add a bit to the scale to avoid having to fix
                # collision detection to ignore contact
                position = midpoint - heading * scale * (1 + epsilon)

                yield (position[0], position[1], theta)
        else:
            raise NotImplementedError(
                "Don't know how to sample from <{}>".format(manifold))

    def cost(self, parent, child):
        """Compute the cost to move between configurations

        Args:
            parent (dict): maps object names to poses
            child (dict): maps object names to poses

        Returns:
            float: the cost to move between configurations

        Raises:
            KeyError if parent is missing an object identifier contained
                in child (if `set(parent.keys()) < set(child.keys())`)
        """
        cost = 0
        for name in child:
            first, second = parent[name], child[name]

            dtheta = math.fabs(first[2] - second[2])
            dtheta = min(dtheta, 2*math.pi-dtheta)

            cost += math.sqrt((first[0]-second[0])**2 +
                              (first[1]-second[1])**2 +
                              dtheta**2)
        return cost

