"""Test dynamics module

Tests are idiosyncratic, because there is no uniform API for dynamics yet.
"""

try:
    import faulthandler
    faulthandler.enable()
except:
    pass

import Box2D.b2 as b2
import shapely.geometry

import metis
from metis.debug import graphical_debug, draw_polygon

def example_geometry():
    """Generate example geometry for testing"""
    get_triangles = metis.geometry.box2d_triangles_from_shapely

    world = b2.world()

    agent = world.CreateDynamicBody()
    agent_geometry = shapely.geometry.Polygon([
        (2./3., 0.), (-1./3., .4), (-1./3., -.4)])
    for triangle in get_triangles(agent_geometry):
        _ = agent.CreateFixture(shape=triangle)

    heptagon = shapely.geometry.Point(0, 0).buffer(1.5, resolution=3)
    offset_hexagon = shapely.geometry.Point(1, 0).buffer(1.5, resolution=2)
    shapes = {
        'box': shapely.geometry.box(0, 0, 1, 1),
        'convex': heptagon,
        'nonconvex': heptagon.difference(offset_hexagon)
        }

    boxes = {}
    for name in shapes:
        boxes[name] = world.CreateDynamicBody()
        for triangle in get_triangles(shapes[name]):
            _ = boxes[name].CreateFixture(shape=triangle)

    return world, agent, boxes

def test_sample_contact_pose():
    """Test ability to sample contact poses

    All sampled poses should place the robot in contact with the target
    object, but without placing them in collision.
    """
    def check_sampled_pose(agent, agent_pose, entity):
        """Check if poses place agent and entity in contact
        """
        get_shape = metis.geometry.shapely_from_box2d_body
        agent_shape = get_shape(agent, agent_pose)
        entity_shape = get_shape(entity)

        tol = 5e-3
        expanded = agent_shape.buffer(tol).intersection(entity_shape.buffer(tol))
        shrunk = agent_shape.buffer(-tol).intersection(entity_shape.buffer(-tol))

        assert shrunk.is_empty, graphical_debug(
            "Objects should not intersect: "
            "they overlap by {0} even when shrunk by {1}".format(shrunk.area, tol),
            lambda ax: draw_polygon(ax, agent_shape, label='agent'),
            lambda ax: draw_polygon(ax, entity_shape, label='entity'))

        assert not expanded.is_empty, graphical_debug(
            "Objects should touch: "
            "they do not overlap even when expanded by {0}".format(tol),
            lambda ax: draw_polygon(ax, agent_shape, label='agent'),
            lambda ax: draw_polygon(ax, entity_shape, label='entity'))
    _, agent, boxes = example_geometry()
    entities = boxes.copy()
    entities['robot'] = agent
    dynamics = metis.dynamics.MagneticDynamics(entities)

    for name, box in boxes.iteritems():
        for sample in dynamics.sample_from_manifold(('robot', name), 100):
            # There's no guarantee that there is only a finite number of
            # sampled poses, so stop after 100
            check_sampled_pose(agent, sample, box)
            # yield check_sampled_pose, agent, sample, box



