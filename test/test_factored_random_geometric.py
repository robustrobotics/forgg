"""Tests for random geometric graphs"""

import Box2D.b2 as b2
import shapely.geometry

import metis

from metis.factored_random_geometric_graphs import (
    FactoredRandomGeometricGraph, NoObjectContactBlacklist)
from metis.debug import graphical_debug, draw_polygon, draw_polygons

def example_world():
    """Create an example Box2D world for testing

    Returns:
        tuple(world, bodies, configuration):
            - world is a Box2D world with multiple dynamic bodies
            - bodies is a dictionary mapping object names to their Box2D
              body
            - configuration is an example collision-free configuration
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

    agent = world.CreateDynamicBody()
    agent_geometry = shapely.geometry.Polygon([
        (2./3., 0.), (-1./3., .4), (-1./3., -.4)])
    for triangle in get_triangles(agent_geometry):
        _ = agent.CreateFixture(shape=triangle)

    boxes = [world.CreateDynamicBody() for _ in xrange(2)]
    for box in boxes:
        box.CreateFixture(shape=b2.polygonShape(box=(.8, .8)))

    bodies = {'robot': agent, 'box1': boxes[0], 'box2': boxes[1]}
    sample_configuration = {
        'robot': (1, 2, 0), 'box1': (3, 2, -.2), 'box2': (5, 2.5, 0.1)}

    return world, bodies, sample_configuration

def test_acyclic_chains():
    """Check that the correct number of acyclic chains are generated"""
    names = ['robot', 'box1', 'box2']
    chains = lambda: FactoredRandomGeometricGraph.acyclic_chains(names)

    expected_number = 16
    actual_number = sum(1 for _ in chains())
    assert actual_number == expected_number, \
        "Expected {} chains; actual value was {}".format(
            expected_number, actual_number)

    assert all(
        FactoredRandomGeometricGraph.is_acyclic(chain)
        for chain in chains())

def test_contains():
    """Check that neighbors are computed correctly"""
    world, bodies, _ = example_world()

    geometry = metis.geometry.ManyShapeGeometry(world, bodies)
    dynamics = metis.dynamics.MagneticDynamics(bodies)
    factored_graph = FactoredRandomGeometricGraph(
        geometry, dynamics, default_count=5, blacklist=NoObjectContactBlacklist())

    assert {'robot': (None, 0), 'box1': (None, 0), 'box2': (None, 0)} in factored_graph
    assert {'robot': ('box1', 0), 'box1': (None, 0), 'box2': (None, 0)} in factored_graph
    assert {'robot': ('box1', 5), 'box1': (None, 0), 'box2': (None, 0)} not in factored_graph
    assert {'robot': ('robot', 0), 'box1': (None, 0), 'box2': (None, 0)} not in factored_graph
    assert {'robot': ('box1', 0), 'box1': ('robot', 0), 'box2': (None, 0)} not in factored_graph

def test_iter():
    """Check if we can iterate over all vertices"""
    world, bodies, _ = example_world()

    geometry = metis.geometry.ManyShapeGeometry(world, bodies)
    dynamics = metis.dynamics.MagneticDynamics(bodies)
    factored_graph = FactoredRandomGeometricGraph(
        geometry, dynamics, default_count=5, blacklist=NoObjectContactBlacklist())

    assert sum(1 for _ in factored_graph) == len(factored_graph), \
        "__iter__ should be consistent with __len__"
    for vertex in factored_graph:
        assert vertex in factored_graph, \
            "__iter__ should be consistent with __contains__"

def test_getitem():
    """Check if we can compute poses from vertices"""
    world, bodies, _ = example_world()

    geometry = metis.geometry.ManyShapeGeometry(world, bodies)
    dynamics = metis.dynamics.MagneticDynamics(bodies)
    factored_graph = FactoredRandomGeometricGraph(
        geometry, dynamics, default_count=5,
        blacklist=NoObjectContactBlacklist())

    vertices = [
        {'robot': (None, 0), 'box1': (None, 0), 'box2': (None, 0)},
        {'robot': ('box1', 0), 'box1': (None, 0), 'box2': (None, 0)},]

    for vertex in vertices:
        configuration = factored_graph[vertex]
        assert all(name in configuration for name in factored_graph.names)
        assert all(len(pose) == 3 for pose in configuration.itervalues())

def test_nearest():
    """Check that we can look up the nearest vertex"""
    world, bodies, sample_configuration = example_world()

    geometry = metis.geometry.ManyShapeGeometry(world, bodies)
    dynamics = metis.dynamics.MagneticDynamics(bodies)
    configurations = {(name, None): [value,]
                      for name, value in sample_configuration.iteritems()}
    factored_graph = FactoredRandomGeometricGraph(
        geometry, dynamics, default_count=100,
        blacklist=NoObjectContactBlacklist(), configurations=configurations)

    vertex = factored_graph.nearest(sample_configuration)
    assert vertex in factored_graph
    assert geometry.configuration_is_free(factored_graph[vertex])
    nearest_configuration = factored_graph[vertex]
    assert nearest_configuration.keys() == sample_configuration.keys()
    for name in sample_configuration:
        assert all(nearest_configuration[name] == sample_configuration[name])

    factored_graph = FactoredRandomGeometricGraph(
        geometry, dynamics, default_count=100, blacklist=NoObjectContactBlacklist())

    vertex = factored_graph.nearest(sample_configuration)
    assert vertex in factored_graph
    assert geometry.configuration_is_free(factored_graph[vertex])
    nearest_configuration = factored_graph[vertex]
    assert nearest_configuration.keys() == sample_configuration.keys()

def test_neighbors():
    """Check that neighbors are computed correctly"""
    world, bodies, sample_configuration = example_world()

    geometry = metis.geometry.ManyShapeGeometry(world, bodies)
    dynamics = metis.dynamics.MagneticDynamics(bodies)
    configurations = {(name, None): [value,]
                      for name, value in sample_configuration.iteritems()}
    factored_graph = FactoredRandomGeometricGraph(
        geometry, dynamics, default_count=100,
        blacklist=NoObjectContactBlacklist(), configurations=configurations)

    vertex = factored_graph.nearest(sample_configuration)
    neighbors = list(factored_graph.neighbors(vertex))

    duplicates = set()
    seen = set()
    for neighbor in neighbors:
        if neighbor in seen:
            duplicates.add(neighbor)
        else:
            seen.add(neighbor)
    assert len(duplicates) == 0, (
        "Neighbors should be unique: had duplicate elements"
        "\n\t".join(str(d) for d in duplicates)
        )

    assert any(v['robot'][0] is not None for v in neighbors), graphical_debug(
        "There should be at least one neighbor with an object in its grasp",
        lambda ax: draw_polygons(ax, {
            name: metis.geometry.shapely_from_box2d_body(bodies[name], pose)
            for name, pose in factored_graph[vertex].iteritems()}))

    grasping = next(v for v in neighbors if v['robot'][0] is not None)
    neighbors = list(factored_graph.neighbors(grasping))
    assert any(v['robot'][0] is None for v in neighbors), graphical_debug(
        "There should be at least one neighbor without an object in its grasp",
        lambda ax: draw_polygons(ax, {
            name: metis.geometry.shapely_from_box2d_body(bodies[name], pose)
            for name, pose in factored_graph[grasping].iteritems()}))
