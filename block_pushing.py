"""Experiment 2: block pushing

This experiment is intended to verify 1) completeness and optimality,
2) show that the factored algorithm is faster, and 3) show both
algorithms obtain better plans than a sequential task and motion
planner.

If there are N objects, there are 2^(N(N-1)/2) modes (every pair of
objects can be in contact or not). We can represent this using an
adjacency graph. However, for simplicity we assume that *only* one block
can move at a time; this removes all but N+1 modes (one for each way to
grasp an object).

Two configurations are on the same orbit if every object not in the same
connected component as the robot has the same location. This can be
represented by a hash: hash the adjacency graph and add the hash of the
location of each object not in the connected component. For the
simplified dynamics, this is just the hash of each object not grasped.

We can sample configurations from an orbit by sampling configurations of
the connected component and assuming the remaining configurations are
constant.

Two modes are adjacent iff at most one object enters or leaves the
connected component of the adjacency graph containing the robot between
the two modes. The intersection of an orbit and a mode is the set of
configurations in which the connected component touches the object (if
the connected component grows) or the configuration of the connected
component (if it shrinks).

We compare three approaches:
    - a conventional 'task then motion' planner, which chooses a
      sequence of blocks to move and moves them into position
    - a Hauser-like algorithm that samples each orbit independently
    - a WRVB algorithm that samples from factors independently

We assume throughout that it is *impossible* to push two blocks at once.
This evens the playing field, since only the algorithms under
consideration could even consider such an option.
"""
# pylint: disable=no-member

import time
import pickle
import shelve
from itertools import chain

import numpy
import Box2D.b2 as b2
import shapely.geometry
import matplotlib.pyplot as pyplot
import metis

# Rename long names for convenience
from metis.debug import draw_polygon, draw_configuration
from metis.factored_random_geometric_graphs import (
    FactoredRandomGeometricGraph, NoObjectContactBlacklist, apply_transform,
    SE2Manifold, apply_inverse_transform)
from metis.geometry import shapely_from_box2d_body as shapely_from_body
from metis.geometry import box2d_triangles_from_shapely as \
    triangles_from_shapely
from metis.geometry import convex_box2d_shape_from_shapely as \
    convex_from_shapely

class Timer:
    """Context manager for timing"""
    def __init__(self, callback=None):
        self.callback = callback
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.callback is not None:
            self.callback(self.interval)

    def elapsed(self):
        """Return time elapsed since timer started"""
        return time.time() - self.start if self.start is not None else None

class Statistics(object):
    """Store algorithmic statistics"""
    def __init__(self):
        super(Statistics, self).__init__()
        self.stats = {}
        self.last_called = {}
        self.counts = {}

    def timer(self, name):
        """Create a context-managing timer

        Timer will automatically append its time to statistics
        """
        return Timer(lambda dt: self.stats.__setitem__(name, dt))

    def __getitem__(self, key):
        return self.stats.__getitem__(key)

    def __setitem__(self, key, value):
        return self.stats.__setitem__(key, value)

    def __delitem__(self, key):
        return self.stats.__delitem__(key)

    def format(self, string):
        """Format string to print statistics"""
        return string.format(**self.stats)

class ProblemDomain(object):
    def __init__(self, upper_door=True):
        """Initialize block-pushing domain"""
        # TODO: make it possible to customize geometry
        super(ProblemDomain, self).__init__()
        # Outer walls
        obstacle_geometry = shapely.geometry.box(0, 0, 10, 10)
        obstacle_geometry = obstacle_geometry.difference(
            obstacle_geometry.buffer(-.2))
        # Central wall
        obstacle_geometry = obstacle_geometry.union(
            shapely.geometry.LineString([(5, 0), (5, 10)])
            .buffer(.1, cap_style=2))
        # Lower door
        obstacle_geometry = obstacle_geometry.difference(
            shapely.geometry.Point(5, 2.5).buffer(1, cap_style=1))
        # Upper door
        if upper_door:
            obstacle_geometry = obstacle_geometry.difference(
                shapely.geometry.Point(5, 7.5).buffer(1, cap_style=1))
        self.obstacle_geometry = obstacle_geometry

        self.agent_geometry = shapely.geometry.Polygon(
            [(2./3., 0.), (-1./3., .4), (-1./3., -.4)])

        self.object_geometry = {
            'box1': shapely.geometry.box(-.5, -.5, .5, .5),
            'box2': shapely.geometry.box(-.5, -.5, .5, .5)}

        self.world, self.agent, self.bodies = self.create_world()
        self.dynamics = metis.dynamics.MagneticDynamics(self.bodies)

    def create_world(self):
        """Initialize the c structures"""
        world = b2.world()
        obstacles = world.CreateStaticBody()
        for triangle in triangles_from_shapely(self.obstacle_geometry):
            _ = obstacles.CreateFixture(shape=triangle)

        agent = world.CreateDynamicBody()
        for triangle in triangles_from_shapely(self.agent_geometry):
            _ = agent.CreateFixture(shape=triangle)

        bodies = {'robot': agent}
        for obj, shape in self.object_geometry.iteritems():
            body = world.CreateDynamicBody()
            body.CreateFixture(shape=convex_from_shapely(shape))
            bodies[obj] = body
        return world, agent, bodies

    def create_many_shape_geometry(self, **kwargs):
        """Create a ManyShapeGeometry object from this instance

        Args:
            **kwargs: additional keyword args to pass to the
                ManyShapeGeometry constructor
        """
        return metis.geometry.ManyShapeGeometry(
            self.world, self.bodies,
            self.obstacle_geometry.bounds, **kwargs)

    def create_se2_geometry(self, fixed_poses, relative_poses, **kwargs):
        """Create an SE2Geometry object from this instance

        Objects that do not appear in either fixed_poses or
        relative_poses are not added to the geometry.

        Args:
            fixed_poses (dict): map from object names to absolute poses;
                objects in this dict are assumed to be fixed in the
                geometry object
            relative_poses (dict): map from object names to relative
                poses; objects in this dict are assumed to be fixed
                relative to the transforms sampled from the geometry
                object
            **kwargs: additional keyword args to pass to the
                SE2Geometry constructor
        """
        obstacles = self.obstacle_geometry.union(shapely.ops.cascaded_union([
            shapely_from_body(self.bodies[body], pose)
            for body, pose in fixed_poses.iteritems()]))

        # agent = shapely_from_body(self.bodies['robot'], (0, 0, 0))
        agent = shapely.ops.cascaded_union([
            shapely_from_body(self.bodies[body], pose)
            for body, pose in relative_poses.iteritems()])

        world = b2.world()
        static_body = world.CreateStaticBody()
        for triangle in triangles_from_shapely(obstacles):
            _ = static_body.CreateFixture(shape=triangle)

        dynamic = world.CreateDynamicBody()
        for triangle in triangles_from_shapely(agent):
            _ = dynamic.CreateFixture(shape=triangle)

        return metis.geometry.SE2Geometry(
            world, dynamic.fixtures, bounds=self.obstacle_geometry.bounds,
            **kwargs)

    def __getstate__(self):
        return (self.obstacle_geometry,
                self.agent_geometry,
                self.object_geometry)

    def __setstate__(self, state):
        self.obstacle_geometry = state[0]
        self.agent_geometry = state[1]
        self.object_geometry = state[2]

        self.world, self.agent, self.bodies = self.create_world()
        self.dynamics = metis.dynamics.MagneticDynamics(self.bodies)

    def draw(self, robot=(2, 2, 0), box1=(3, 5, -.2), box2=(5, 2.5, 0.1)):
        """Draw the problem domain"""
        sample_configuration = {
            'robot': robot,
            'box1': box1,
            'box2': box2}
        axes = pyplot.subplot(111, aspect='equal')
        draw_polygon(axes, self.obstacle_geometry)
        draw_configuration(axes, self.bodies, sample_configuration)
        axes.autoscale(True)
        pyplot.show()

class ProblemInstance(object):
    """A problem instance (cost, start state, goal)"""
    def __init__(self, domain, robot=(2, 2, 0), box1=(3, 5, -.2),
                 box2=(5, 2.5, 0.1)):
        super(ProblemInstance, self).__init__()
        self.domain = domain
        self.world = domain.world
        self.bodies = domain.bodies
        self.dynamics = domain.dynamics
        self.start = {
            'robot': robot,
            'box1': box1,
            'box2': box2}

        self.goal = {'box1': shapely.geometry.box(5, 0, 10, 5)}

    def is_goal(self, configuration):
        """True if configuration satisfies the goal"""
        return all(region.contains(shapely_from_body(self.bodies[obj],
                                                     configuration[obj]))
                   for (obj, region) in self.goal.iteritems())

    def heuristic(self, configuration):
        """Returns a lower bound on the cost to reach the goal"""
        # pylint: disable=unused-argument,no-self-use
        return 0

    def cost_estimate(self, parent, child):
        """Returns a cheap estimate of the cost to move between
        configurations"""
        return self.dynamics.cost(parent, child)

    def __getstate__(self):
        return (self.domain, self.start, self.goal)

    def __setstate__(self, state):
        domain = state[0]
        self.domain = domain
        self.world = domain.world
        self.bodies = domain.bodies
        self.dynamics = domain.dynamics

        self.start = state[1]
        self.goal = state[2]

class Orbit(object):
    def __init__(self, mode, configuration):
        """
        Args:
            mode (str): the dynamical mode on which this orbit lies,
                represented as the name of the object being grasped, or
                'None' if no object is grasped.
            configuration (dict): a full configuration chosen from this
                orbit, represented as a map from object names to (x, y,
                theta) tuples
        """
        self.mode = mode
        self.configuration = configuration
        self.geometry = geometry

    def __eq__(self, other):
        return self.holding == other.holding and all(
            self.configuration[o] == other.configuration[o]
            for o in self.configuration if o != 'robot' and o != self.holding)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """Compute a hash that respects the equivalence class of orbits

        Two orbits are equivalent if all objects but the robot and its
        cargo (if any) are at the same location.
        """
        return hash((self.holding,) + tuple(sorted(
            self.configuration[o] for o in self.configuration
            if o != 'robot' and o != self.holding)))

    def sample_configurations(self, count):
        """Sample count configurations from this orbit
        """
        pass

    def adjacent_orbits(self):
        """Generate all orbits which may intersect this orbit"""
        if self.mode is not None:
            yield None
        else:
            for obj in self.configuration:
                if obj != 'robot':
                    yield obj

    def sample_from_intersection(self, mode, count):
        """Sample count configurations from the intersection of this
        orbit and another mode"""
        if self.mode is None and mode in self.configuration:
            # we are sampling grasps of an object
            pass
        elif self.mode is not None and mode is None:
            # we are sampling from the intersection of this orbit with
            # the 'nongrasping' mode: yield any configuration on this
            # orbit but tagged with the nongrasping mode
            pass
        else:
            raise ValueError(("Modes not adjacent", mode, self.mode))

class Solver(object):
    def __init__(self, instance, log):
        super(Solver, self).__init__()
        self.instance = instance
        self.log = log

class OBT(Solver):
    """Perform bookkeeping for the generic-orbit (Hauser) algorithm
    """
    def __init__(self, instance, log, count=200, epsilon=0., eta=1., seed=0):
        super(OBT, self).__init__(instance, log)
        self.count = count
        self.epsilon = epsilon
        self.eta = eta
        self.seed = seed

class TAMP(Solver):
    def __init__(self, instance, log, count=200, epsilon=0., eta=1., seed=0):
        super(TAMP, self).__init__(instance, log)
        self.count = count
        self.epsilon = epsilon
        self.eta = eta
        self.seed = seed

    def configure_subproblem(self, start_description, goal_description):
        """Configure a subproblem for solving

        A subproblem, in this context, is a graph to search and a
        Boolean function that determines which vertices are goal
        vertices.
        """
        # pylint: disable=too-many-arguments,too-many-locals
        start, start_holding, holding_pose = start_description
        goal_region, end_holding = goal_description

        fixed_poses = {'box1': start['box1'],
                       'box2': start['box2']}
        relative_poses = {'robot': (0, 0, 0)}

        # If we are holding something at start, set that now
        if start_holding is not None:
            assert holding_pose is not None
            del fixed_poses[start_holding]
            relative_poses[start_holding] = holding_pose
        self.log.info(
            "{} // {}".format(fixed_poses.keys(), relative_poses.keys()))
        geometry = self.instance.domain.create_se2_geometry(
            fixed_poses=fixed_poses, relative_poses=relative_poses,
            seed=self.seed)

        configurations = [start['robot']]
        if end_holding is not None:
            goal_rpos = [
                tf for tf in self.instance.dynamics.sample_from_manifold(
                    ('robot', end_holding), None)]
            configurations += [
                apply_transform(start[end_holding], tf) for tf in goal_rpos]
            goal_vertices = set(xrange(1, len(configurations)))
        else:
            goal_vertices = set()
            goal_rpos = None

        graph = metis.random_geometric_graphs.SE2RandomGraph(
            geometry, self.count, epsilon=self.epsilon, eta=self.eta,
            configurations=configurations)

        is_goal = lambda vertex: (
            vertex in goal_vertices or
            (goal_region is not None and
             graph.geometry.fixtures_in_region(graph[vertex], goal_region)))

        # from_fixtures = metis.geometry.shapely_from_box2d_fixtures
        # obstacles = from_fixtures(
        #     f for body in geometry.world for f in body.fixtures
        #     if f not in geometry.fixtures)
        # for goal_vertex in goal_vertices:
        #     cfg = configurations[goal_vertex]
            # metis.debug.graphical_debug("goal is {}free".format(
            #     '' if geometry.configuration_is_free(cfg) else 'not '),
            #     lambda ax, c=cfg: draw_polygon(
            #         ax, from_fixtures(geometry.fixtures, c), fc='r'),
            #     lambda ax: draw_polygon(ax, obstacles))
        return graph, is_goal, goal_rpos

    def search(self, graph, is_goal):
        """Search a graph

        Performs Dijkstra's algorithm on graph with inital vertex 0 and
        goal criterion is_goal
        """
        start_vertex = 0
        goal_vertex = None

        open_set = metis.queue.PriorityQueue([(start_vertex, 0)])
        cost_to_come = {start_vertex: 0}
        parent = {start_vertex: None}
        seen = set()

        pop_count = 0
        with Timer() as timer:
            while open_set:
                pop_count += 1
                if pop_count % 500 == 0:
                    self.log.info("after {dt:.2f} s, {n} vertices explored "
                                  "({vps:.2f} vertices per second)"
                                  .format(dt=timer.elapsed(), n=pop_count,
                                          vps=pop_count/timer.elapsed()))

                current = open_set.pop()
                seen.add(current)
                if is_goal(current):
                    goal_vertex = current
                    break
                for neighbor in graph.neighbors(current):
                    if neighbor in seen:
                        continue
                    new_cost = (cost_to_come[current] +
                                graph.cost(current, neighbor))
                    if new_cost < cost_to_come.get(neighbor, float('inf')):
                        parent[neighbor] = current
                        cost_to_come[neighbor] = new_cost
                        open_set[neighbor] = new_cost
            self.log.info("{count} vertices explored in {time} seconds".format(
                count=pop_count, time=timer.elapsed()))

            return {'graph': graph,
                    'search_count': pop_count,
                    'start_vertex': start_vertex,
                    'goal_vertex': goal_vertex,
                    'search_time': timer.elapsed(),
                   }, {'parent': parent, 'cost_to_come': cost_to_come}

    def prepare_results(self, start_description, goal_description,
                        result, extra):
        """Prepare detailed results"""
        start, start_holding, holding_pose = start_description
        _, end_holding = goal_description
        goal_vertex = result['goal_vertex']
        cost_to_come = extra['cost_to_come']
        parent = extra['parent']
        goal_rpos = result['goal_rpos']

        if goal_vertex is not None:
            self.log.info("Found path to goal with cost {}".format(
                cost_to_come[goal_vertex]))
            vertex_path = [goal_vertex]
            while parent[vertex_path[-1]] is not None:
                vertex_path.append(parent[vertex_path[-1]])
            vertex_path.reverse()

            # Unpack the path
            path = []
            for vertex in vertex_path:
                state = {}
                for obj in ('robot', 'box1', 'box2'):
                    if obj == 'robot':
                        state[obj] = result['graph'][vertex]
                    elif obj == start_holding:
                        state[obj] = apply_transform(
                            result['graph'][vertex], holding_pose)
                    else:
                        state[obj] = start[obj]
                path.append(state)
            path_cost = cost_to_come[goal_vertex]
            end = path[-1]
            assert vertex_path[0] == result['start_vertex']
        else:
            self.log.info("Found no path to goal")

            vertex_path = None
            path = None
            path_cost = float('inf')
            end = None

        if end_holding is None:
            rpos = None
        elif goal_vertex is not None and goal_vertex <= len(goal_rpos):
            rpos = goal_rpos[goal_vertex-1]
        else:
            rpos = None

        return goal_vertex is not None, {
            'start': start,
            'end': end,
            'path': path,
            'path_cost': path_cost,
            'goal_rpos': rpos,
        }

    def solve_subproblem(self, start_description, goal_description):
        """Solve a subproblem

        Args:
            start_description (tuple): three element tuple describing start
                dict: initial configuration at start of subproblem
                str: name of object the robot is holding at the start of
                    subproblem, or None if not holding anything
                tuple: relative pose of held object if not None
            goal_description (tuple): two element tuple describing goal
                shapely geometry: goal achieved when robot/object pair is in
                    goal_region
                str: name of object the robot is to be holding
                at the end of subproblem, or None if not to hold
                anything
        """
        # Default is not holding anything at start
        graph, is_goal, goal_rpos = self.configure_subproblem(
            start_description, goal_description)
            # (start, start_holding, holding_pose), (goal_region, end_holding))

        result, extra = self.search(graph, is_goal)
        result['goal_rpos'] = goal_rpos

        success, prepared = self.prepare_results(
            start_description, goal_description, result, extra)
        result.update(prepared)
        return success, result, extra

    def try_plan(self, plan, plan_name):
        """Try to instantiate a plan"""
        results = {
            'component_paths': [],
            'steps': [],
            'path_cost': 0,
            'search_time': 0,
            'search_count': 0,
            }
        extras = []
        start = dict(self.instance.start)
        holding_pose = None
        for j, step in enumerate(plan):
            step_name = "{}_step{}".format(plan_name, j)
            with Timer() as timer:
                goal_region, start_holding, end_holding = step
                assert start_holding is None or holding_pose is not None

                start_description = (start, start_holding, holding_pose)
                goal_description = (goal_region, end_holding)
                success, result, extra = self.solve_subproblem(
                    start_description, goal_description)
                result['search_time'] = timer.elapsed()

                results['component_paths'].append(result['path'])
                results['steps'].append(result)
                results['path_cost'] += result['path_cost']
                results['search_count'] += result['search_count']
                results['search_time'] += result['search_time']

                extras.append(extra)

                # Prepare for next iteration
                start = result['end']
                if result['goal_rpos'] is not None:
                    holding_pose = apply_inverse_transform(
                        numpy.array((0, 0, 0)), result['goal_rpos'])
                else:
                    holding_pose = None
            self.log.info(
                "{} completed {} ({} vertices explored in {} seconds".format(
                    step_name, 'successfully' if success else 'unsuccessfully',
                    result['search_count'], result['search_time']))
            if not success:
                return False, results, extras
        results['path'] = list(chain.from_iterable(
            step for step in results['component_paths']))
        return True, results, extras

    def solve(self):
        """Solves a problem instance using a simple TAMP solver

        Rather than implement a PDDL planner and a description of the
        problem, we hard code several reasonable plans and then instantiate
        them. The first to return successfully is returned as the solution.
        """
        # Now, hard code a few plans and try to instantiate them
        corner = shapely.geometry.box(0, 7, 3, 10)
        goal = self.instance.goal['box1'] # TODO: too much hardcoding
        plans = [
            [(None, None, 'box1'), # No goal region, grasp box1
             (goal, 'box1', None)], # Move box1 to goal region
            [(None, None, 'box2'), # Grasp box2
             (corner, 'box2', None), # Move box2 to goal region
             # (corner, None, None), # drop box2d
             (None, None, 'box1'), # Grasp box1
             (goal, 'box1', None)], # Move box1 to goal region
        ]
        results = {
            'plans': [],
            'search_time': 0,
            'search_count': 0,
            }
        extras = {}
        success = False
        for i, plan in enumerate(plans):
            plan_name = "plan{}".format(i)
            success, result, extra = self.try_plan(plan, plan_name)
            results['plans'].append(result)
            extras[str(i)] = extra
            results['search_count'] += result['search_count']
            results['search_time'] += result['search_time']
            if success:
                results['path_cost'] = result['path_cost']
                results['path'] = result['path']
                results['component_paths'] = result['component_paths']
                break
        if not success:
            results['path_cost'] = float('inf')
            results['path'] = None
            results['component_paths'] = None
        return success, results, extras

class FORGG(Solver):
    def __init__(self, instance, log, count=200, epsilon=0., eta=1., seed=0,
                 maxcount=None):
        super(FORGG, self).__init__(instance, log)
        self.count = count
        self.epsilon = epsilon
        self.eta = eta
        self.seed = seed
        self.maxcount = maxcount

        self.statistics = Statistics()

        with self.statistics.timer('build_time') as timer:
            self.geometry = metis.geometry.ManyShapeGeometry(
                instance.world, instance.bodies,
                instance.domain.obstacle_geometry.bounds, seed=seed)
            configurations = {(name, None): [value,]
                              for name, value in instance.start.iteritems()}
            self.factored_graph = FactoredRandomGeometricGraph(
                self.geometry, instance.dynamics, default_count=count,
                blacklist=NoObjectContactBlacklist(),
                configurations=configurations)
        self.statistics['build_count'] = len(self.factored_graph)
        self.log.info(self.statistics.format(
            "{build_time} seconds elapsed\ngraph has {build_count} nodes"))

    def solve(self):
        """Solves a problem instance using a simple TAMP solver

        Rather than implement a PDDL planner and a description of the
        problem, we hard code several reasonable plans and then instantiate
        them. The first to return successfully is returned as the solution.
        """
        instance = self.instance
        start_vertex = self.factored_graph.nearest(instance.start)
        goal_vertex = None

        open_set = metis.queue.PriorityQueue(
            [(start_vertex, instance.heuristic(start_vertex))])
        cost_to_come = {start_vertex: 0}
        parent = {start_vertex: None}
        seen = set()

        goal_vertex = None
        with self.statistics.timer('search_time') as timer:
            pop_count = 0
            while open_set:
                pop_count += 1
                if self.maxcount is not None and pop_count > self.maxcount:
                    break
                elif pop_count % 1000 == 0:
                    self.log.info("after {dt:.2f} s, {n} vertices explored "
                                  "({vps:.2f} vertices per second)"
                                  .format(dt=timer.elapsed(), n=pop_count,
                                          vps=pop_count/timer.elapsed()))

                current = open_set.pop()
                seen.add(current)
                if instance.is_goal(self.factored_graph[current]):
                    goal_vertex = current
                    break
                for neighbor in self.factored_graph.neighbors(current):
                    if neighbor in seen:
                        continue
                    new_cost = (cost_to_come[current] +
                                self.factored_graph.cost(current, neighbor))
                    if new_cost < cost_to_come.get(neighbor, float('inf')):
                        parent[neighbor] = current
                        cost_to_come[neighbor] = new_cost
                        open_set[neighbor] = new_cost
            self.statistics['search_count'] = pop_count
        self.log.info(self.statistics.format(
            "{search_count} vertices explored in {search_time} seconds"))

        if goal_vertex is not None:
            self.log.info("Found path to goal with cost {}".format(
                cost_to_come[goal_vertex]))
            vertex_path = [goal_vertex]
            while parent[vertex_path[-1]] is not None:
                vertex_path.append(parent[vertex_path[-1]])
            vertex_path.reverse()

            path = [self.factored_graph[v] for v in vertex_path]
            path_cost = cost_to_come[goal_vertex]
            assert vertex_path[0] == start_vertex
        else:
            self.log.info("Found no path to goal")
            vertex_path = None
            path = None
            path_cost = float('inf')

        return goal_vertex is not None, {
            'graph': self.factored_graph,
            'start_vertex': start_vertex,
            'goal_vertex': goal_vertex,
            'search_time': self.statistics['search_time'],
            'search_count': self.statistics['search_count'],
            'build_time': self.statistics['build_time'],
            'build_count': self.statistics['build_count'],
            'vertex_path': vertex_path,
            'path': path,
            'path_cost': path_cost
        }, {'parent': parent, 'cost_to_come': cost_to_come}


def run(task_name, log, parameters):
    """Run a block pushing task"""
    algorithm = parameters['algorithm']

    domain = ProblemDomain(**parameters['domain'])
    instance = ProblemInstance(domain, **parameters['instance'])
    try:
        if algorithm == 'forgg':
            success, result, extra = FORGG(
                instance, log, **parameters['solver']).solve()
        elif algorithm == 'tamp':
            success, result, extra = TAMP(
                instance, log, **parameters['solver']).solve()
        else:
            log.info('Unknown algorithm {}'.format(algorithm))
            return False
        log.info("\nResults:" + '\n'.join(
            "    {}: {}".format(field, value)
            for (field, value) in sorted(result.iteritems())
            if isinstance(value, (int, long, float))))

        filename = task_name + '.p'
        with open(filename, 'wb') as handle:
            data = {'instance': instance, 'result': result}
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        shelf = shelve.open(task_name + '.s')
        shelf.update(extra)
        shelf.close()

        return success
    except MemoryError:
        log.exception("Search failed: out of memory")
        return False
