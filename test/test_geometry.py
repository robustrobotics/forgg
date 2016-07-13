"""Tests for geometry module"""

import numpy
import Box2D.b2 as b2
import shapely.geometry
import shapely.ops

import metis
from metis.debug import graphical_debug, draw_polygon, draw_polygons


def example_shapes():
    """Generate example shapes for testing"""
    obstacle_geometry = shapely.geometry.box(0, 0, 10, 10)
    obstacle_geometry = obstacle_geometry.difference(obstacle_geometry.buffer(-.2))
    obstacle_geometry = obstacle_geometry.union(
        shapely.geometry.LineString([(5, 0), (5, 9)]).buffer(.1, cap_style=2))
    return {
        'square': shapely.geometry.Polygon(
            [(1, 1), (-1, 1), (-1, -1), (1, -1)]),
        'monotonic': shapely.geometry.Polygon(
            [(1, 2), (-1, 2), (-3, 1), (-3, -1), (-1, -2), (1, -2), (0, -1),
             (0, 1), (1, 2)]),
        'star': shapely.geometry.Polygon(
            [(0, 1), (-1, 0.5), (-2, 0.7), (-1.5, 0), (-2, -0.5), (0, -0.3),
             (2, -0.5), (1.5, 0), (2, 0.7), (1, 0.5), (0, 1)]),
        'holes': shapely.geometry.box(0, 0, 2, 2).difference(
            shapely.geometry.box(.5, .5, 1.5, 1.5)),
        'split_holes': shapely.geometry.box(0, 0, 2, 2).difference(
            shapely.geometry.box(.3, .3, 1.7, 1.7)).union(
                shapely.geometry.box(.9, 0, 1.1, 2)),
        'almost_split_holes': shapely.geometry.box(0, 0, 2, 2).difference(
            shapely.geometry.box(.1, .1, 1.9, 1.9)).union(
                shapely.geometry.box(.9, 0, 1.1, 1.7)),
        'pathological': shapely.geometry.Polygon(
            [(0, 0), (5, 0), (5, 3), (2, 3), (1, 1), (1, 3), (0, 3)],
            [[(1.9, 2), (2, 2.5), (2.5, 2), (2, 1.5)],
             [(3, 2), (3.5, 2.5), (4, 2), (3.5, 1.5)]]),
        'multipolygon': shapely.geometry.box(0, 0, 1, 1).union(
            shapely.geometry.box(2, 0, 3, 1)),
        'obstacle': obstacle_geometry
    }

def test_sparsify_point_set():
    """Check if sparsify returns a list of the specified length"""
    old_array = numpy.random.random_sample((20, 3))
    new_array = metis.geometry.sparsify_point_set(old_array, 10)
    assert len(new_array) == 10

def test_triangulate():
    """Generate tests of triangulation routine"""
    for name, shape in example_shapes().iteritems():
        yield check_triangulate, name, shape

def check_triangulate(name, shape):
    """Check if reconstructing a triangulation yields the original shape"""
    triangles = metis.geometry.triangulate(shape)
    reconstructed = shapely.ops.cascaded_union(triangles)
    difference = reconstructed.symmetric_difference(shape).area
    assert difference < 1e-6, graphical_debug(
        "triangulation failed for {} (error {})".format(name, difference),
        lambda ax: draw_polygon(ax, shape, label='shape'),
        lambda ax: draw_polygon(ax, reconstructed, label='reconstructed'))

def test_conversion():
    """Generate tests of shapely/box2d conversion functions"""
    for name, shape in example_shapes().iteritems():
        yield check_exact_conversion, name, shape
        yield check_convex_conversion, name, shape

        if name in {'square', 'multipolygon'}:
            yield check_direct_conversion, name, shape

def check_exact_conversion(name, shape):
    """Check if exact conversion between shapely and box2d works as expected"""
    b2_shapes = metis.geometry.box2d_triangles_from_shapely(shape)
    reconstructed = shapely.ops.cascaded_union([
        metis.geometry.shapely_from_box2d_shape(b2_shape)
        for b2_shape in b2_shapes])
    difference = reconstructed.symmetric_difference(shape).area
    assert difference < 1e-5, graphical_debug(
        "exact conversion failed for {} (error {})".format(name, difference),
        lambda ax: draw_polygon(ax, shape, label='shape'),
        lambda ax: draw_polygon(ax, reconstructed, label='reconstructed'))

def check_convex_conversion(name, shape):
    """Check conversion of convex hull between shapely and box2d"""
    b2_shape = metis.geometry.convex_box2d_shape_from_shapely(shape)
    reconstructed = metis.geometry.shapely_from_box2d_shape(b2_shape)
    convex_hull = shape.convex_hull
    difference = reconstructed.symmetric_difference(convex_hull).area
    assert difference < 1e-6, graphical_debug(
        "convex conversion failed for {} (error {})".format(name, difference),
        lambda ax: draw_polygon(ax, shape, label='shape'),
        lambda ax: draw_polygon(ax, reconstructed, label='reconstructed'))

def check_direct_conversion(name, shape):
    """Check direct conversion between shapely and box2d for convex shapes"""
    b2_shapes = metis.geometry.box2d_shapes_from_shapely(shape)
    reconstructed = shapely.ops.cascaded_union([
        metis.geometry.shapely_from_box2d_shape(b2_shape)
        for b2_shape in b2_shapes])
    difference = reconstructed.symmetric_difference(shape).area
    assert difference < 1e-6, graphical_debug(
        "reconstruction failed for {} (error {})".format(name, difference),
        lambda ax: draw_polygon(ax, shape, label='shape'),
        lambda ax: draw_polygon(ax, reconstructed, label='reconstructed'))

def example_world():
    """Generate a simple example world for testing"""
    convex_from_shapely = metis.geometry.convex_box2d_shape_from_shapely

    obstacle_geometry = shapely.geometry.box(0, 0, 2, 2).difference(
        shapely.geometry.box(.1, .1, 1.9, 1.9))
    moveable_geometry = shapely.geometry.box(-.1, -.1, .1, .1)

    world = b2.world()
    obstacles = world.CreateStaticBody()
    for triangle in metis.geometry.triangulate(obstacle_geometry):
        obstacles.CreateFixture(shape=convex_from_shapely(triangle))

    moveable = world.CreateDynamicBody()
    moveable.CreateFixture(shape=convex_from_shapely(moveable_geometry))

    return world, obstacles, moveable

def test_point_free():
    """Test if point_free performs as expected"""
    world, _, _ = example_world()
    assert metis.geometry.point_free(world, (.5, .5))
    assert not metis.geometry.point_free(world, (.05, .05))

def test_segment_free():
    """Test if segment_free performs as expected"""
    world, _, _ = example_world()
    assert metis.geometry.segment_free(world, (.5, .5), (1, 1))
    assert not metis.geometry.segment_free(world, (.5, .5), (2, 2))
    assert not metis.geometry.segment_free(world, (0, 0), (1, 1))

def test_pose_free():
    """Generate the actual tests for pose_free"""
    yield check_pose_free, "clear", True, (0.3, 0.3, 0)
    yield check_pose_free, "rotated", True, (2.09, 2.09, numpy.pi/4)
    yield check_pose_free, "collision", False, (0, 0, 0)

def test_path_free():
    """Generate the actual tests for path_free"""
    yield check_path_free, "clear", True, (0.4, 0.4, 0), (.6, .6, 0.5)
    yield check_path_free, "start_collision", False, (0, 0, 0), (.5, .5, 0)
    yield check_path_free, "end_collision", False, (1, 1, 0), (2, 2, 0)

def check_pose_free(name, expected, pose):
    """Test if pose_free performs as expected"""
    pose_free = metis.geometry.pose_free
    world, obstacles, moveable = example_world()
    shapes = {
        'obstacles': metis.geometry.shapely_from_box2d_body(obstacles),
        'moveable': metis.geometry.shapely_from_box2d_body(moveable, pose)}
    actual = pose_free(world, moveable.fixtures, pose)
    assert expected == actual, graphical_debug(
        "case {}: pose_free(world, moveable.fixtures, {}) was {};"
        "expected {}".format(name, pose, actual, expected),
        lambda ax: draw_polygons(ax, shapes))

def check_path_free(name, expected, start, end):
    """Test if path_free performs as expected"""
    pose_free = metis.geometry.pose_free
    world, obstacles, moveable = example_world()
    shapes = {
        'obstacles': metis.geometry.shapely_from_box2d_body(obstacles),
        'start': metis.geometry.shapely_from_box2d_body(moveable, start),
        'end': metis.geometry.shapely_from_box2d_body(moveable, end)}
    actual = pose_free(world, moveable.fixtures, start, end)
    assert expected == actual, graphical_debug(
        "case {}: pose_free(world, moveable.fixtures, {}, {}) was {};"
        "expected {}".format(name, start, end, actual, expected),
        lambda ax: draw_polygons(ax, shapes))

def test_r2geometry():
    """Test if point geometry performs as expected"""
    world, obstacles, moveable = example_world()
    geometry = metis.geometry.R2Geometry(world)
    for _ in xrange(100):
        start = geometry.sample_configuration()
        end = geometry.sample_configuration()
        assert len(start) == 2
        assert geometry.configuration_is_free(start)

        assert len(end) == 2
        assert geometry.configuration_is_free(end)

        shapes = {
            'obstacles': metis.geometry.shapely_from_box2d_body(obstacles),
            'moveable': metis.geometry.shapely_from_box2d_body(moveable),
            'segment': shapely.geometry.LineString([start, end])
        }

        # path_is_free should be reflexive
        forward_free = geometry.path_is_free(start, end)
        backward_free = geometry.path_is_free(end, start)
        assert forward_free == backward_free

        # If both points are in the interior of the world, the path
        # should be free
        interior = lambda x, y: 0 < x < 2 and 0 < y < 2
        if interior(*start) and interior(*end):
            assert forward_free, graphical_debug(
                "Expected path from {} to {} to be free".format(start, end),
                lambda ax: draw_polygons(ax, shapes))

    assert 3. < geometry.mu_free < 4.

def test_se2geometry():
    """Test if shape geometry performs as expected"""
    world, obstacles, moveable = example_world()
    bounds = metis.geometry.world_bounding_box(
        world, ignore_fixtures=moveable.fixtures)
    geometry = metis.geometry.SE2Geometry(
        world, moveable.fixtures, bounds=bounds)
    for _ in xrange(100):
        start = geometry.sample_configuration()
        end = geometry.sample_configuration()
        assert len(start) == 3
        assert geometry.configuration_is_free(start)

        assert len(end) == 3
        assert geometry.configuration_is_free(end)

        shapes = {
            'obstacles': metis.geometry.shapely_from_box2d_body(obstacles),
            'start': metis.geometry.shapely_from_box2d_body(moveable, start),
            'end': metis.geometry.shapely_from_box2d_body(moveable, end),
        }

        # path_is_free should be reflexive
        forward_free = geometry.path_is_free(start, end)
        backward_free = geometry.path_is_free(end, start)
        assert forward_free == backward_free

        # Because the free space is convex, all configurations should be
        # in the interior of the world and hence all paths should be
        # free
        assert forward_free, graphical_debug(
            "Expected path from {} to {} to be free".format(start, end),
            lambda ax: draw_polygons(ax, shapes))

def test_multiobjectgeometry():
    convex_from_shapely = metis.geometry.convex_box2d_shape_from_shapely

    obstacle_geometry = shapely.geometry.box(0, 0, 2, 2).difference(
        shapely.geometry.box(.1, .1, 1.9, 1.9))
    moveable_geometry = shapely.geometry.box(-.1, -.1, .1, .1)

    world = b2.world()
    obstacles = world.CreateStaticBody(userData="obstacles")
    for triangle in metis.geometry.triangulate(obstacle_geometry):
        obstacles.CreateFixture(shape=convex_from_shapely(triangle))

    box1 = world.CreateDynamicBody(userData="box1")
    box1.CreateFixture(shape=convex_from_shapely(moveable_geometry))

    box2 = world.CreateDynamicBody(userData="box2")
    box2.CreateFixture(shape=convex_from_shapely(moveable_geometry))

    geometry = metis.geometry.ManyShapeGeometry(
        world, {"box1": box1, "box2": box2})

    yield check_multiobject_configuration, "free", True, geometry, {
        'box1': (.5, .5, 0), 'box2': (1.5, 1.5, 0)}
    yield check_multiobject_configuration, "rotated", True, geometry, {
        'box1': (.5, .5, numpy.pi/4), 'box2': (.65, .65, numpy.pi/4)}
    yield check_multiobject_configuration, "obstacle_collision", False, \
        geometry, {'box1': (0, 0, 0), 'box2': (1.5, 1.5, 0)}
    yield check_multiobject_configuration, "moveable_collision", False, \
        geometry, {'box1': (.5, .5, 0), 'box2': (.6, .6, 0)}

    yield (check_multiobject_path, "free", True, geometry,
           {'box1': (.5, .5, 0), 'box2': (1.5, 1.5, 0)},
           {'box1': (.5, 1.5, 0), 'box2': (1.5, .5, 0)})

    yield (check_multiobject_path, "collision", False, geometry,
           {'box1': (.5, .5, 0), 'box2': (1.5, 1.5, 0)},
           {'box1': (-1.5, 1.5, 0), 'box2': (1.5, .5, 0)})

    yield (check_multiobject_path, "missed_collision", True, geometry,
           {'box1': (.5, .5, 0), 'box2': (1.5, 1.5, 0)},
           {'box1': (1.5, 1.5, 0), 'box2': (.5, .5, 0)})

    yield (check_multiobject_path, "caught_collision", False, geometry,
           {'box1': (.5, .5, 0), 'box2': (1.5, 1.5, 0)},
           {'box1': (1.5, 1.5, 0)})

def check_multiobject_configuration(name, expected, geometry, configuration):
    """Check if multiobject geometry computes free configurations correctly"""
    actual = geometry.configuration_is_free(configuration)
    shapes = shapely.ops.cascaded_union([
        metis.geometry.shapely_from_box2d_body(body)
        for body in geometry.world.bodies])
    assert expected == actual, graphical_debug(
        "case {}: geometry.configuration_is_free({}) was {};"
        "expected {}".format(name, configuration, actual, expected),
        lambda ax: draw_polygon(ax, shapes, label='shapes'))

def check_multiobject_path(name, expected, geometry, parent, child):
    """Check if multiobject geometry computes free paths correctly"""
    actual = geometry.path_is_free(parent, child)
    shapes = shapely.ops.cascaded_union([
        metis.geometry.shapely_from_box2d_body(body)
        for body in geometry.world.bodies])
    children = shapely.ops.cascaded_union([
        metis.geometry.shapely_from_box2d_body(geometry.bodies[name], pose)
        for name, pose in child.iteritems()])
    assert expected == actual, graphical_debug(
        "case {}: geometry.path_is_free({}, {}) was {};"
        "expected {}".format(name, parent, child, actual, expected),
        lambda ax: draw_polygon(ax, shapes, label='shapes'),
        lambda ax: draw_polygon(ax, children, label='children'))

