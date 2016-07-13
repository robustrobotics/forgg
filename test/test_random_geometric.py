"""Tests for random geometric graphs"""

import Box2D.b2 as b2
import shapely.geometry

import metis
from metis.debug import graphical_debug, draw_polygon, draw_undirected_r2graph

def draw_box2d(ax, world):
    draw_polygon(ax, metis.geometry.shapely_from_box2d_world(world))

def example_world():
    """Create an example Box2D world for testing

    Returns:
        tuple(world, robot):
            - world is a Box2D world with a dynamic body
            - robot is a Box2D body describing a robot in the world
    """
    get_triangles = metis.geometry.box2d_triangles_from_shapely

    obstacle_geometry = shapely.geometry.box(0, 0, 10, 10)
    obstacle_geometry = obstacle_geometry.difference(
        obstacle_geometry.buffer(-.2))
    obstacle_geometry = obstacle_geometry.union(
        shapely.geometry.LineString([(5, 0), (5, 10)]).buffer(.1, cap_style=2))
    obstacle_geometry = obstacle_geometry.difference(
        shapely.geometry.Point(5, 2.5).buffer(1, cap_style=1))
    obstacle_geometry = obstacle_geometry.difference(
        shapely.geometry.Point(5, 7.5).buffer(1, cap_style=1))

    world = b2.world()
    obstacles = world.CreateStaticBody()
    for triangle in get_triangles(obstacle_geometry):
        _ = obstacles.CreateFixture(shape=triangle)

    return world

def test_euclidean_random_graph():
    """Test random disc graph"""
    world = example_world()
    count = 200

    geometry = metis.geometry.R2Geometry(world)
    graph = metis.random_geometric_graphs.EuclideanRandomGraph(
        geometry, count, eta=2)

    assert graph.is_connected(), graphical_debug(
        "graph should be connected",
        lambda ax: draw_box2d(ax, world),
        lambda ax: draw_undirected_r2graph(ax, graph))

    assert len(graph) == count, graphical_debug(
        "number of vertices was {0} (expected {1})".format(len(graph), count),
        lambda ax: draw_box2d(ax, world),
        lambda ax: draw_undirected_r2graph(ax, graph))

    label = 0
    assert label in graph, \
        "label {0} should be in graph".format(label)
    assert geometry.configuration_is_free(graph[label]), \
        "pose with label {0} should be free".format(label)

def test_se2_random_graph():
    """Test random disc graph in SE(2)"""
    world = example_world()
    count = 200

    dynamic = world.CreateDynamicBody()
    agent = shapely.geometry.Polygon(
        [(2./3., 0.), (-1./3., .4), (-1./3., -.4)])
    for triangle in metis.geometry.box2d_triangles_from_shapely(agent):
        _ = dynamic.CreateFixture(shape=triangle)

    geometry = metis.geometry.SE2Geometry(world, dynamic.fixtures,
                                          bounds=(0, 0, 10, 10), seed=0)
    graph = metis.random_geometric_graphs.SE2RandomGraph(
        geometry, count, eta=2)

    assert graph.is_connected(), graphical_debug(
        "graph should be connected",
        lambda ax: draw_box2d(ax, world))

    assert len(graph) == count, graphical_debug(
        "number of vertices was {0} (expected {1})".format(len(graph), count),
        lambda ax: draw_box2d(ax, world))

    label = 0
    assert label in graph, \
        "label {0} should be in graph".format(label)
    assert geometry.configuration_is_free(graph[label]), \
        "pose with label {0} should be free".format(label)

def test_delaunay_random_graph():
    """Test random planar graph"""
    world = example_world()
    count = 100

    geometry = metis.geometry.R2Geometry(world)
    graph = metis.random_geometric_graphs.RandomPlanarGraph(geometry, count)

    assert graph.is_connected(), graphical_debug(
        "graph should be connected",
        lambda ax: draw_box2d(ax, world),
        lambda ax: draw_undirected_r2graph(ax, graph))

    assert len(graph) == count, graphical_debug(
        "number of vertices was {0} (expected {1})".format(len(graph), count),
        lambda ax: draw_box2d(ax, world),
        lambda ax: draw_undirected_r2graph(ax, graph))

    label = 0
    assert label in graph, \
        "label {0} should be in graph".format(label)
    assert geometry.configuration_is_free(graph[label]), \
        "pose with label {0} should be free".format(label)

