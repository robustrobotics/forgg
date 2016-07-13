"""Tools for defining and modeling problem geometry

This module implements several kinds of collision detection, as well
as tools for converting between different representations of problem
geometry. The supported geometry types are referred to as
:math:`\\mathbf{R}^2` (where the robot is a point navigating in a
polygonal world), :math:`\\text{SE}(2)` (where the robot is a polygon
navigating in a polygonal world), and many shape (where the robot is a
polygon and the world contains other dynamic polygonal objects, which
can be pushed or pulled).

In addition to standard tools like ``numpy``, it relies on three
libraries:

- shapely_: a (mostly) pure python library for constructive and
  set-theoretic geometry. It lets you define objects *constructively*::

      walls = shapely.geometry.box(0, 0, 2, 2).difference(
          shapely.geometry.box(.1, .1, 1.9, 1.9))

  which defines a rectangular room by subtracting a square with side
  length 1.8 from a square of side length 2. It also lets you reason
  about things like boundaries and intersections of regions, and then
  recover the edges, vertices, etc of the shapes you create.  Compared
  to Box2D it is slow, easy to use, and reliable.
- Box2D_: a ``python`` wrapper based on ``SWIG`` around a ``c++``
  library for 2D physics simulation. It is extremely fast compared to
  shapely, but also buggy, hard to use, and not overly feature-rich.  As
  soon as it's feasible I will remove Box2D from this package, but
  irritating as it is, it gets the job done. Many of the functions in
  this module are for translating Box2D objects to shapely geometry and
  vice versa.
- triangle_: a ``cython`` wrapper around a very fast, very stable ``c``
  library for triangulation.  I use it for breaking non-convex shapes
  into convex pieces, since Box2D can only handle convex geometry. It is
  used exclusively in the ``triangulate`` method.

To make it easier to tell which objects are shapely objects and which
are Box2D objects, I try to follow a consistent naming convention:

- ``point`` is always a tuple or tuple-like containing coordinates (x, y)
  on the 2D plane
- ``pose`` is always a tuple or tuple-like containing a pose in SE(2),
  represented as (x, y, theta), with (x, y) a position in meters and
  theta an angle in radians
- ``world`` is always a Box2D world, which contains a collection of ``bodies``
  defining the objects (both static and dynamic) in the world
- ``body`` is always a Box2D body, which models an object and contains
  both a pose (a position and orientation) as well as collection of
  fixtures which define the geometry of the object.
- ``fixture`` is always a Box2D fixture, which contains a convex polygon
  and is a component of the geometry of a ``body``
- ``geometry`` is a shapely geometry object; this may be a point, line,
  polygon, or collection of polygons, but will always define a region of
  the 2D plane.

.. _shapely: http://toblerity.org/shapely/manual.html
.. _Box2D: https://github.com/pybox2d/pybox2d/wiki/manual
.. _triangle: http://dzhelil.info/triangle/
"""
import math

import numpy
import scipy.spatial
import triangle
import Box2D.b2 as b2
import shapely.geometry
import shapely.ops
# pylint: disable=no-member

from metis.iterators import circular_pairs

def sparsify_point_set(points, count):
    """Downsample a list of points

    Reduce a list of points to length count, attempting to minimize the
    maximum distance from a point in the original set to a point in the
    new set.  This is done using an O(M^2 N) algorithm, where M=count
    and N=len(points); this could be easily improved to O(M (log M + log
    N) + N log N) using a more complex algorithm.

    Args:
        points (numpy.ndarray): array of points. Each row of the array
            is treated as a point, so the width of the array is the
            dimensionality of the sample space.
        count (int): number of points to keep

    Returns:
        numpy.ndarray: downsampled array of points

    Examples:
        >>> sparsify_point_set([(0, 0), (1, 1), (2, 2)], 2)
        array([[0, 0],
               [2, 2]])
    """
    points = numpy.array(points)
    if points.shape[0] < count:
        return points
    else:
        mask = [0]
        while len(mask) < count:
            dist = scipy.spatial.distance.cdist(points, points[mask, ...])
            mask.append(numpy.argmax(numpy.amin(dist, 1)))
        return points[mask, ...]

def triangulate(geometry, max_area=None):
    """Use the triangle library to triangulate a polygon

    Args:
        polygon (shapely.*): shapely geometry representing the area to
            triangulate
        max_area (float): If provided, the triangulation will be refined
            until all triangles have area less than this maximum.

    Returns:
        list: list of triangular polygons
    """
    if not geometry.is_valid:
        raise ValueError("Tried to triangulate invalid geometry", geometry)

    if hasattr(geometry, "geoms"): # polygon is a MultiPolygon
        polygons = list(geometry.geoms)
    else:
        polygons = [geometry]

    vertices = []
    segments = []
    for polygon in polygons:
        offset = len(vertices)
        vertices.extend(polygon.exterior.coords[0:-1])
        segments.extend(circular_pairs(
            range(offset, offset+len(polygon.exterior.coords)-1)))
        for ring in polygon.interiors:
            offset = len(vertices)
            vertices.extend(ring.coords[0:-1])
            segments.extend(circular_pairs(
                range(offset, offset+len(ring.coords)-1)))

    shape = {'vertices': numpy.array(vertices),
             'segments': numpy.array(segments, dtype=numpy.int32)}

    # Find the holes in the geometry
    buffer_by = numpy.sqrt(geometry.envelope.area)
    complement = geometry.envelope.buffer(
        buffer_by, cap_style=2, join_style=2).difference(geometry)
    if complement.geom_type == "MultiPolygon":
        shape['holes'] = numpy.array([interior.representative_point().coords[0]
                                      for interior in complement.geoms])
    elif complement.geom_type == "Polygon":
        shape['holes'] = numpy.array(complement.representative_point().coords[0])

    if max_area is None:
        opts = "p"
    else:
        opts = "pa{}".format(max_area)

    triangulation = triangle.triangulate(shape, opts)
    return [shapely.geometry.Polygon([triangulation['vertices'][i]
                                      for i in triplet])
            for triplet in triangulation['triangles']]

def box2d_shapes_from_shapely(geometry):
    """Create Box2D shapes from shapely geometry"""
    return [b2.polygonShape(vertices=list(poly.exterior.coords))
            for poly in getattr(geometry, "geoms", [geometry])]

def convex_box2d_shape_from_shapely(geometry):
    """Create a Box2D shape from the convex hull of shapely geometry"""
    return b2.polygonShape(
        vertices=list(geometry.convex_hull.exterior.coords))

def box2d_triangles_from_shapely(geometry):
    """Create Box2D shapes by triangulating shapely geometry"""
    for face in triangulate(geometry):
        yield b2.polygonShape(vertices=list(face.exterior.coords))

def shapely_from_box2d_shape(shape):
    """Create shapely geometry from a Box2D shape"""
    return shapely.geometry.polygon.orient(shapely.geometry.Polygon(
        shape.vertices))

def shapely_from_box2d_fixtures(fixtures, pose=None):
    """Create shapely geometry from a list of Box2D fixtures"""
    transform = b2.transform()
    transform.position = pose[0:2] if pose is not None else (0, 0)
    transform.angle = pose[2] if pose is not None else 0
    # TODO: ensure polys are oriented
    return shapely.ops.cascaded_union([
        shapely.geometry.Polygon([
            transform  * p for p in fixture.shape.vertices])
        for fixture in fixtures])

def shapely_from_box2d_body(body, pose=None):
    """Create shapely geometry from a Box2D body"""
    if pose is None:
        return shapely.geometry.polygon.orient(shapely.ops.cascaded_union([
            shapely.geometry.Polygon([
                body.GetWorldPoint(p) for p in fixture.shape.vertices])
            for fixture in body.fixtures]))
    else:
        transform = b2.transform()
        transform.position = pose[0:2]
        transform.angle = pose[2]
        return shapely.geometry.polygon.orient(shapely.ops.cascaded_union([
            shapely.geometry.Polygon([
                transform  * p for p in fixture.shape.vertices])
            for fixture in body.fixtures]))

def shapely_from_box2d_world(world):
    """Create shapely geometry from a Box2D shape"""
    return shapely.geometry.polygon.orient(shapely.ops.cascaded_union([
        shapely.geometry.Polygon([
            body.GetWorldPoint(p) for p in fixture.shape.vertices])
        for body in world.bodies
        for fixture in body.fixtures]))

def bounding_box(fixtures, pose):
    """Get the axis aligned bounding box of the fixtures

    Args:
        fixtures (iterable): an iterable containing the fixtures to bound
        pose (tuple): an (x, y, theta) tuple. All fixtures will be
            transformed by this pose before the bounding box is
            computed.

    Returns:
        b2.aabb: the smallest axis-aligned bounding box which completely
        encloses all fixtures after being transformed by pose. If no
        fixtures are supplied, the value None (rather than an empty
        bounding box) is returned.
    """
    transform = b2.transform()
    transform.position = pose[0:2]
    transform.angle = pose[2]

    aabb = None
    for fixture in fixtures:
        # 0 is the 'child index', which is not described in the pybox2d
        # documentation so I'm not really sure what it is.
        if aabb is None:
            aabb = fixture.shape.getAABB(transform, 0)
        else:
            aabb.Combine(fixture.shape.getAABB(transform, 0))
    return aabb

def world_bounding_box(world, ignore_fixtures=None):
    """Get the smallest bounds tuple which encloses all fixtures

    Args:
        world (b2.world): box2d geometry to query
        ignore_fixtures (iterable): if provided, the fixtures to ignore
            when computing the bounding box

    Returns:
        tuple: the smallest (xmin, xmax, ymin, ymax) which completely
            encloses all fixtures except those in ignore_fixtures. If
            there are no fixtures in the world, raise an error
    """
    aabb = shapely.geometry.Point()
    for body in world.bodies:
        for fixture in body.fixtures:
            if ignore_fixtures is not None and fixture in ignore_fixtures:
                continue
            else:
                shape = shapely.geometry.Polygon(
                    [body.GetWorldPoint(p) for p in fixture.shape.vertices])
                aabb = aabb.union(shape).envelope
    return None if aabb.is_empty else aabb.bounds

class PointFreeCallback(b2.queryCallback):
    """Callback class for Box2D shape collision detection

    Used to determine if a point is in free space.

    The existence of this class is an artifact of pybox2d's poor
    design; it uses callback classes rather than callbacks, which
    make sense in ``c++`` but are silly in python.

    Attributes:
        collision_free (bool): the return value of the callback. After
            being queried, this will be True if the path is collision
            free.
        point (tuple): point to test
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, point):
        b2.queryCallback.__init__(self)
        self.collision_free = True
        self.point = point

    def ReportFixture(self, fixture):
        """Check if the query point is inside this fixture"""
        # pylint: disable=invalid-name
        if fixture.TestPoint(self.point):
            self.collision_free = False
        return self.collision_free

class RayCastCallback(b2.rayCastCallback):
    """Callback class for Box2D point collision detection

    Used to determine if the straight-line path between two points
    is collision free in a Box2D world.

    The existence of this class is an artifact of pybox2d's poor
    design; it uses callback classes rather than callbacks, which
    make sense in ``c++`` but are silly in python.

    Attributes:
        collision_free (bool): the return value of the callback.
            After being queried, this will be True if the path is
            collision free.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self):
        b2.rayCastCallback.__init__(self)
        self.collision_free = True

    def ReportFixture(self, fixture, point, normal, fraction):
        """If the function is called, there is at least one collision"""
        # pylint: disable=invalid-name,unused-argument
        self.collision_free = False
        return 0

class ShapeFreeCallback(b2.queryCallback):
    """Callback class for Box2D shape collision detection

    Used to determine if the straight-line path between two poses of a
    shape is collision free in a Box2D world. Angles are interpolated
    linearly, so for large rotations this will be inaccurate.

    The test is done by casting each of the test_fixtures (one at at
    time) at each fixture in the queried AABB that is not in the whitelist.

    The existence of this class is an artifact of pybox2d's poor design;
    it uses callback classes rather than callbacks, which make sense in
    ``c++`` but are silly in python.

    Attributes:
        test_fixtures (list): list of fixtures to check for collision
        whitelist (list): list of fixtures to skip in collision
            detection
        pose_a (tuple): start pose of the path as an (x, y, theta) tuple
        pose_b (tuple): end pose of the path as an (x, y, theta) tuple
        collision_free (bool): the return value of the callback. After
            being queried, this will be True if the path is collision
            free.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, test_fixtures, pose_a, pose_b=None, whitelist=None):
        b2.queryCallback.__init__(self)
        self.test_fixtures = test_fixtures
        self.whitelist = whitelist if whitelist is not None else []
        self.pose_a = pose_a
        self.pose_b = pose_b if pose_b is not None else pose_a
        self.collision_free = True

    def ReportFixture(self, other_fixture):
        """Check for collisions

        This function is called by Box2D during an AABB query for each
        fixture in the AABB.
        """
        # pylint: disable=invalid-name
        if other_fixture in self.whitelist:
            return True
        if other_fixture in self.test_fixtures:
            # This ignores self-collisions (which seems a silly thing to
            # have to check for...)
            # TODO: there should be a way to avoid this check
            return True

        sweep = b2.sweep(a0=self.pose_a[2], a=self.pose_b[2],
                         c0=self.pose_a[0:2], c=self.pose_b[0:2])
        still = b2.sweep(a=other_fixture.body.angle,
                         a0=other_fixture.body.angle,
                         c=other_fixture.body.position,
                         c0=other_fixture.body.position)
        for test_fixture in self.test_fixtures:
            (collision, impact_time) = b2.timeOfImpact(
                shapeA=test_fixture.shape, shapeB=other_fixture.shape,
                sweepA=sweep, sweepB=still, tMax=1.)
            self.collision_free &= (
                (impact_time == 1.) and
                (collision == b2.TOIOutput.e_separated or
                 collision == b2.TOIOutput.e_touching))
            if not self.collision_free:
                break
        return self.collision_free

def point_free(world, point):
    """Check if a point intersects any object in a Box2D world

    Args:
        world (b2.world): a Box2D world containing the geometry to query.
        point (tuple): an (x, y) tuple defining the query point.

    Returns:
        bool: True if there are no objects in world within .1mm of
        point; False otherwise.
    """
    query = PointFreeCallback(point)
    aabb = b2.AABB()
    aabb.lowerBound = (point[0]-1e-4, point[1]-1e-4)
    aabb.upperBound = (point[0]+1e-4, point[1]+1e-4)
    world.QueryAABB(query, aabb)
    return query.collision_free

def segment_free(world, point_a, point_b):
    """Check if a line segment intersects any object in a Box2D world

    Args:
        world (b2.world): a Box2D world containing the geometry to query.
        point_a (tuple): an (x, y) tuple defining the start of the
            segment
        point_b (tuple): an (x, y) tuple defining the end of the segment

    Returns:
        bool: True if the line segment does not intersect any object in
        the world
    """
    if not point_free(world, point_a):
        return False
    else:
        callback = RayCastCallback()
        world.RayCast(callback, point_a, point_b)
        return callback.collision_free

def pose_free(world, fixtures, pose_a, pose_b=None, whitelist=None):
    """Deterimine if a Box2D world is collision free

    Used to determine if the straight-line path between two poses of a
    shape is collision free in a Box2D world. Angles are interpolated
    linearly, so for large rotations this will be inaccurate.

    Attributes:
        world (b2.world): Box2D object describing world geometry
        fixtures (sequence): an iterable sequence of Box2D fixtures
            to ignore in collision detection. Typically this is a
            triangulation of the moveable object being queried.
        pose_a (tuple): an (x, y, theta) tuple defining the pose for
            which to perform collision detection
        pose_b (tuple): if provided, check for collision along a linear
            sweep from pose_a to pose_b, much like segment_free.
        whitelist (sequence): an iterable sequence of Box2D fixtures to
            *ignore* when performing collision detection. Useful for
            preventing Box2D from reporting a collision between objects
            in contact (say, if the robot is grasping an object)

    Returns:
        bool: True if the fixtures are not in collision when transformed
            by pose_a, or never in collision during a linear sweep from
            pose_a to pose_b
    """
    # assert fixtures is not None and pose_a is not None
    aabb = bounding_box(fixtures, pose_a)
    if pose_b is not None:
        aabb.Combine(bounding_box(fixtures, pose_b))
    query = ShapeFreeCallback(fixtures, pose_a, pose_b, whitelist=whitelist)
    world.QueryAABB(query, aabb)
    return query.collision_free

class Geometry(object):
    """Encapsulates collision detection

    This is a base class or prototype for concrete instantiations of
    geometry objects. There are three key things such objects should be
    capable of doing:

        - check whether configurations are collision free
        - check whether there exists a simple collision-free path
          between configurations
        - provide sufficient information to sample random configurations

    Strictly speaking, a base class like this is unnecessary in
    ``python``, but I think it's helpful for readability to lay out the
    interface in one place.
    """
    def __init__(self, seed=0):
        super(Geometry, self).__init__()
        self.random = numpy.random.RandomState(seed)

    def configuration_is_free(self, configuration):
        """Check if the supplied configuration is free"""
        pass

    def path_is_free(self, parent, child):
        """Check if there is a trivial path between configurations"""
        pass

    def sample_configuration(self):
        """Sample a configuration uniformly from the space"""
        pass

class R2Geometry(Geometry):
    """Two dimensional point geometry

    The configuration space is a subset of the 2D Cartesian plane in a
    world that may contain obstacles.

    Attributes:
        world (Box2D.b2world): Underlying Box2D world object
        bounds (tuple): a tuple (xmin, ymin, xmax, ymax) defining the
            bounds of the problem geometry. Used primarily for sampling.
        sample_count (int): number of samples generated so far. Used to
            compute the measure of the set of free configurations
        rejection_count (int): number of samples rejected so far. Used
            to compute the measure of the set of free configurations.
    """
    def __init__(self, world, bounds=None, **kwargs):
        super(R2Geometry, self).__init__(**kwargs)
        self.world = world

        if bounds is None:
            self.bounds = world_bounding_box(world)
        else:
            self.bounds = bounds
        self.sample_count = 0
        self.rejection_count = 0
        self.scale = numpy.array([1, 1])

    @property
    def mu_free(self):
        """The Lebesgue measure of the space of free poses"""
        xmin, ymin, xmax, ymax = self.bounds
        total_area = (xmax - xmin) * (ymax - ymin)
        assert self.rejection_count <= self.sample_count
        if self.sample_count > 0:
            free_count = self.sample_count - self.rejection_count
            fraction_free = float(free_count) / float(self.sample_count)
            return total_area * fraction_free
        else:
            return total_area

    def configuration_is_free(self, point):
        """Check if a point is in the space of free configurations

        Args:
            point (tuple): a point in R2 represented as a tuple (x, y)

        Returns:
            bool: True if ``point`` lies in the free space
        """
        return point_free(self.world, point)

    def path_is_free(self, parent, child):
        """Check if the line segment between two points is in free space

        Args:
            parent (tuple): a point in R2, represented as a tuple
            child (tuple): a point in R2, represented as a tuple

        Returns:
            bool: True if the line segment connecting parent and child
            is in free space
        """
        return segment_free(self.world, parent, child)

    def sample_configuration(self):
        """Sample a configuration uniformly from the free space"""
        xmin, ymin, xmax, ymax = self.bounds
        iteration = 0
        max_iterations = 100
        while iteration < max_iterations:
            point = self.random.uniform(0, 1, size=2)
            point[0] *= xmax - xmin
            point[0] += xmin
            point[1] *= ymax - ymin
            point[1] += ymin

            self.sample_count += 1
            if self.configuration_is_free(point):
                return point
            else:
                self.rejection_count += 1
                iteration += 1
        raise RuntimeError(
            "Failed to return a sample after {} iterations."
            "Problem might be infeasible.".format(iteration))

class SE2Geometry(Geometry):
    """Two dimensional shape geometry

    The configuration space is the space of poses of a 2D shape in a
    world that may contain obstacles.

    Attributes:
        world (Box2D.b2world): Underlying c world object
        fixtures (list): List of fixture objects constituting the
            moving object. A fixture is a shape defined relative to the
            moving object origin.
        bounds (tuple): a tuple (xmin, ymin, xmax, ymax) defining the
            bounds of the problem geometry. Used primarily for sampling.
        sample_count (int): number of samples generated. Used to compute
            the measure of the set of free configurations
        rejection_count (int): number of samples rejected. Used to
            compute the measure of the set of free configurations.
        scale (numpy.array): vector used to scale configurations for
            distance computations. In SE(2), the relative scale of the
            position and rotation components is arbitrary; changing
            units of distance from meters to kilometers, or of rotation
            from radians to degrees, would cause large differences in a
            nearest-neighbors search. The scale property provides a
            suggestion for how to scale the distances between
            configurations, which can be used in random graph
            construction.
    """
    def __init__(self, world, fixtures, bounds=None, **kwargs):
        super(SE2Geometry, self).__init__(**kwargs)
        self.world = world
        self.fixtures = fixtures

        if bounds is None:
            self.bounds = world_bounding_box(world)
        else:
            self.bounds = bounds
        self.sample_count = 0
        self.rejection_count = 0
        # Angle differences will be scaled by this in computing the
        # distance between points in SE(2)
        rotation_scale = 1
        self.scale = numpy.array([1, 1, rotation_scale])

    @property
    def mu_free(self):
        """The Lebesgue measure of the space of free poses"""
        xmin, ymin, xmax, ymax = self.bounds
        total_area = (xmax - xmin) * (ymax - ymin)
        mu_sample = total_area * 2 * math.pi * numpy.prod(self.scale)
        assert self.rejection_count <= self.sample_count
        if self.sample_count > 0:
            free_count = self.sample_count - self.rejection_count
            fraction_free = float(free_count) / float(self.sample_count)
            return mu_sample * fraction_free
        else:
            return mu_sample

    def configuration_is_free(self, pose):
        """Check if a pose is in collision

        Args:
            pose (tuple): a pose in SE(2), represented as a tuple (x, y,
                theta), with theta in radians

        Returns:
            bool: True if ``shape`` would lie entirely in free space if
            transformed by ``pose``.
        """
        return pose_free(self.world, self.fixtures, pose)

    def path_is_free(self, parent, child):
        """Check if the path between configurations is collision-free

        Args:
            parent (tuple): a pose in SE(2), represented as a tuple (x, y,
                theta), with theta in radians
            child (tuple): a pose in SE(2), represented as a tuple (x, y,
                theta), with theta in radians

        Returns:
            bool: True if ``shape`` would lie entirely in free space if
            swept along the linear path from an initial transform
            ``parent`` to a final transform ``child``
        """
        return pose_free(self.world, self.fixtures, parent, child)

    def sample_configuration(self):
        """Sample a configuration uniformly from the free space

        Sampling is done (as per usual) using rejection sampling.

        Returns:
            tuple: a collision-free pose (x, y, theta)

        Raises:
            RuntimeError if rejection sampling fails to find a
                collision-free sample within a fixed number of
                iterations.  Typically this indicates that the Lebesgue
                measure of free space is zero or nearly zero, and thus
                that the problem is infeasible
        """
        xmin, ymin, xmax, ymax = self.bounds
        iteration = 0
        max_iterations = 100
        while iteration < max_iterations:
            pose = self.random.uniform(0, 1, size=3)
            pose[0] *= xmax - xmin
            pose[0] += xmin
            pose[1] *= ymax - ymin
            pose[1] += ymin
            pose[2] *= 2 * math.pi
            pose[2] -= math.pi

            self.sample_count += 1
            if self.configuration_is_free(pose):
                return pose
            else:
                self.rejection_count += 1
                iteration += 1
        raise RuntimeError(
            "Failed to return a sample after {} iterations."
            "Problem might be infeasible.".format(iteration))

    def fixtures_in_region(self, transform, region):
        """Check if the fixtures lie within the given region"""
        return region.contains(
            shapely_from_box2d_fixtures(self.fixtures, transform))

class ManyShapeGeometry(Geometry):
    """Use Box2D to model 2D geometry

    The configuration space is the space of poses of several 2D shape in a
    world that may contain obstacles; it is represented internally as a
    dictionary mapping object names to object poses.

    Attributes:
        world (Box2D.b2world): Underlying Box2D world object
        empty_world (Box2D.b2world): A copy of the underlying world
            object with all dynamic objects removed.
        bodies (dict): a map from object names to Box2D bodies
            describing object geometry
        bounds (tuple): a tuple (xmin, ymin, xmax, ymax) defining the
            bounds of the problem geometry. Used primarily for sampling.
        sample_count (int): number of samples generated. Used to compute
            the measure of the set of free configurations
        rejection_count (int): number of samples rejected. Used to
            compute the measure of the set of free configurations.
        scale (numpy.array): vector used to scale configurations for
            distance computations. In SE(2), the relative scale of the
            position and rotation components is arbitrary; changing
            units of distance from meters to kilometers, or of rotation
            from radians to degrees, would cause large differences in a
            nearest-neighbors search. The scale property provides a
            suggestion for how to scale the distances between
            configurations, which can be used in random graph
            construction.
    """
    def __init__(self, world, bodies, bounds=None, **kwargs):
        super(ManyShapeGeometry, self).__init__(**kwargs)
        self.world = world
        self.empty_world = b2.world()
        self.bodies = bodies

        obstacles = self.empty_world.CreateStaticBody()
        moveable_fixtures = {fixture for body in bodies.itervalues()
                             for fixture in body.fixtures}
        for body in world.bodies:
            for fixture in body.fixtures:
                if fixture not in moveable_fixtures:
                    obstacles.CreateFixture(shape=fixture.shape)

        if bounds is None:
            self.bounds = world_bounding_box(world)
        else:
            self.bounds = bounds

        self.sample_count = 0
        self.rejection_count = 0
        # Angle differences will be scaled by this in computing the
        # distance between points in SE(2)
        rotation_scale = 1
        self.scale = numpy.array([1, 1, rotation_scale])

    @property
    def mu_free(self):
        """The Lebesgue measure of the space of free poses

        Note: this is not the theoretically correct mu_free, because it
        would be insane to sample in the full high-dimensional space.
        Instead this is the average free space across all objects. I
        don't yet have any rigorous theory for what mu_free should be
        for factored graphs, but this is a reasonable guess.
        """
        xmin, ymin, xmax, ymax = self.bounds
        total_area = (xmax - xmin) * (ymax - ymin)
        mu_sample = total_area * 2 * math.pi * numpy.prod(self.scale)
        assert self.rejection_count <= self.sample_count
        if self.sample_count > 0:
            free_count = self.sample_count - self.rejection_count
            fraction_free = float(free_count) / float(self.sample_count)
            return mu_sample * fraction_free
        else:
            return mu_sample

    def configuration_is_free(self, configuration, skip_static=False):
        """Check if a configuration is collision-free

        Performs collision-checking both against the static geometry and
        against the other movable objects in a reasonably efficient way.

        Args:
            configuration (dict): configuration of the objects,
                represented as a map from the name of each object to its
                pose represented as an (x, y, theta) tuple. It is an
                error if the pose of any object is not specified.
            skip_static (bool): if True, skip checking if the
                configuration is in collision with static geometry and
                only check dynamic geometry. Useful to speed things up
                during planning.

        Returns:
            bool: True if no two objects overlap

        Raises:
            ValueError if configuration does not specify the pose of one
                of the objects
        """
        # First check for collisions between each object and the static
        # world geometry
        for (name, body) in self.bodies.iteritems():
            pose = configuration[name]
            if not (skip_static or
                    pose_free(self.empty_world, body.fixtures, pose)):
                return False
            body.position = pose[0:2]
            body.angle = pose[2]
        # Next, force collision detection between dynamic bodies
        self.world.Step(0, 0, 0)
        for contact in self.world.contacts:
            _, _, distance, _ = b2.distance(
                shapeA=contact.fixtureA.shape,
                shapeB=contact.fixtureB.shape,
                transformA=contact.fixtureA.body.transform,
                transformB=contact.fixtureB.body.transform, useRadii=False)
            if distance <= 1e-4:
                return False
        return True

    def path_is_free(self, parent, child, skip_configuration=False):
        """Check if the path between configurations is free

        The path between configurations is assumed to be a linear sweep.
        This class does not do any detection of dynamic constraints;
        multiple object are free to move at once. This is to maintain
        separation of concerns; the problem of determining if there is
        an action which would lead to the configuration change is
        beyond the scope of geometry.

        Because the usual use-case is after dynamics and collisions
        between moving objects have already been considered, this method
        does *not* consider the possibility of collisions between
        objects whose pose changes between parent and child. That check
        can be performed using the objects_collide method.

        Args:
            parent (dict): initial configuration of the objects,
                represented as a map from the name of each object to its
                pose represented as an (x, y, theta) tuple. It is an
                error if any object's pose is not specified.
            child (dict): final configuration of the objects,
                represented as a map from the name of each object to its
                pose represented as an (x, y, theta) tuple. If an object
                is missing from the dict, its pose is assumed to be the
                same as in the parent configuration.
            skip_configuration (bool): if True, assume the parent
                configuration is collision free without checking. Note
                oddities in Box2D can make this miss collisions if the
                parent configuration *is* in collision.

        Returns:
            bool: True if the trivial path (a linear sweep) between
            poses is collision-free
        """
        if not (skip_configuration or self.configuration_is_free(parent)):
            return False
        else:
            dynamic_fixtures = sum(
                (self.bodies[name].fixtures for name in child), [])
            for name in child:
                if not pose_free(self.world, self.bodies[name].fixtures,
                                 parent[name], child[name],
                                 whitelist=dynamic_fixtures):
                    return False
            return True

    def objects_collide(self, parent, child):
        """Check if objects will collide with each other

        Unlike path_is_free, this method does *not* require a pose for
        all objects, and does *not* consider collisions with objects not
        mentioned in the partial configuration or with static obstacles.
        Instead, it checks if any object in the partial configuration
        collides with any other object as it moves along a linear sweep
        from its initial pose to its final pose.

        Collision checking against specific static objects can be
        coerced by including their pose in ``parent``; if an object does
        not appear in ``child``, its pose is assumed to be the same as in
        ``parent``.

        Args:
            parent (dict): initial configuration of the objects,
                represented as a map from the name of each object to its
                pose represented as an (x, y, theta) tuple. Any object
                not included in the map is ignored, including static
                obstacles.
            child (dict): final configuration of the objects,
                represented as a map from the name of each object to its
                pose represented as an (x, y, theta) tuple. If an object
                is missing from ``child`` but is included in ``parent``, its
                pose is assumed to be the same as in ``parent``

        Returns:
            bool: True if the trivial path (a linear sweep) between
            poses is collision-free
        """
        # I don't need this yet so I won't bother implementing it, but
        # the plan is to create an AABB for each sweep, compare AABBs
        # pairwise (which is quadratic complexity but fast, probably
        # faster than using any n log n structure), then do ToI
        # detection for any pair of objects whose AABBs collide.
        raise NotImplementedError

    def sample_configuration(self):
        """Sample a random configuration for all objects

        Returns:
            dict: a collision-free configuration of all objects,
            represented a as a map from object names to poses,
            themselves represented as (x, y, theta) tuples

        Raises:
            RuntimeError if rejection sampling fails to return a sample
        """
        iteration = 0
        max_iterations = 100
        while iteration < max_iterations:
            configuration = {name: self.sample_object_configuration(name)
                             for name in self.bodies}
            if self.configuration_is_free(configuration):
                return configuration
            else:
                iteration += 1
        raise RuntimeError(
            "Failed to return a sample after {} iterations."
            "Problem might be infeasible.".format(iteration))

    def sample_object_configuration(self, name):
        """Sample a random pose for one object

        Args:
            name (str): the name of the object to sample

        Returns:
            tuple: a collision-free pose of the specified object,
            represented as an (x, y, theta) tuple

        Raises:
            KeyError if name is not the identifier of a body
            RuntimeError if rejection sampling fails to return a sample
        """
        xmin, ymin, xmax, ymax = self.bounds
        iteration = 0
        max_iterations = 100
        while iteration < max_iterations:
            pose = self.random.uniform(0, 1, size=3)
            pose[0] *= xmax - xmin
            pose[0] += xmin
            pose[1] *= ymax - ymin
            pose[1] += ymin
            pose[2] *= 2 * math.pi
            pose[2] -= math.pi

            self.sample_count += 1
            if pose_free(self.empty_world, self.bodies[name].fixtures, pose):
                return pose
            else:
                self.rejection_count += 1
                iteration += 1
        raise RuntimeError(
            "Failed to return a sample after {} iterations."
            "Problem might be infeasible.".format(iteration))

