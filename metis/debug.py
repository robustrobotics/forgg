"""Tools for graphical debugging

These are simple functions to draw the world state when something goes
wrong. They are not intended to make production-ready graphics or to
build a UI around, but instead focus on putting as much useful
information on the screen as possible.
"""

import numpy
import shapely.geometry
import matplotlib.pyplot as pyplot
import descartes.patch as descartes

import metis

def graphical_debug(msg, *args):
    """Open a matplotlib plot to explain failed tests

    Args:
        msg: an explanation for why the plot is being created
        *args: a list of graphical debugging functions to be called.
            Each function is passed a set of axes in which to draw, and
            any return value is ignored.

    Returns:
        msg: function returns whatever was passed as 'msg'. This enables
            a simple assert syntax:
                `assert False, graphical_debug("Explanation")`
            will draw the debug plot only if an assert is triggered, and
            print the `explanation` to the console as an assert usually
            would.

    Examples:
        >>> metis.GRAPHICAL_DEBUGGING_ENABLED = False
        >>> graphical_debug("Example", lambda ax: ax.plot([0],[0]))
        'Example'
    """
    if metis.graphical_debugging_enabled():
        axes = pyplot.axes(aspect='equal')
        axes.set_title(msg)
        for arg in args:
            arg(axes)
        axes.autoscale(True)
        pyplot.show()

    return msg

def draw_polygon(axes, geometry, **kwargs):
    """Draw shapely geometry on the specified axes

    Args:
        axes: matplotlib axes in which to draw geometry
        geometry: a shapely Polygon or MultiPolygon
        kwargs: any keyword args accepted by the constructor for
            matplotlib.patches.Patch. See
            http://matplotlib.org/api/patches_api.html#matplotlib.patches.Patch
            for a list. By default, this function sets `alpha` and
            `facecolor` if unset, and overrides `label` to center the
            specified text on the centroid of the geometry.

    Raises:
        ValueError if the geometry is not a Polygon or a MultiPolygon
    """
    kwargs.setdefault('alpha', .5)
    kwargs.setdefault('facecolor', '#33ff33')
    label = kwargs.pop('label', '')

    if geometry.is_empty:
        return
    elif geometry.geom_type == 'Polygon':
        axes.add_patch(descartes.PolygonPatch(geometry, **kwargs))
    elif geometry.geom_type == 'MultiPolygon':
        for geom in geometry.geoms:
            axes.add_patch(descartes.PolygonPatch(geom, **kwargs))
    else:
        msg = "Unkown geometry type '{}'".format(geometry.geom_type)
        raise ValueError(msg)
    if label is not None:
        centroid = numpy.array(geometry.centroid)
        axes.annotate(label, xy=centroid, xytext=centroid,
                      horizontalalignment='center', verticalalignment='center')

def draw_line(axes, geometry, **kwargs):
    """Draw shapely geometry on the specified axes

    Args:
        axes: matplotlib axes in which to draw geometry
        geometry: a LineString, LinearRing, or Point shapely object
        kwargs: any keyword args accepted by the constructor for
            matplotlib.lines.Line2D for Point, LineString, or LinearRing
            geometries. See
            http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D
            for a list.
    """
    if geometry.geom_type in ['LineString', 'LinearRing', 'Point']:
        axes.plot([p[0] for p in geometry.coords],
                  [p[1] for p in geometry.coords], **kwargs)

def draw_polygons(axes, shapes, keywords=None, **kwargs):
    """Draw many shapely polygons on the specified axes

    Args:
        axes: matplotlib axes in which to draw geometry
        shapes (dict): maps object names to shapely polygons
        keywords (dict): maps object names to **able dicts of keywords
            to be passed to draw_polygon for that object
        kwargs: keyword args to be passed to draw_polygon for each
            object. Valid keywords are those accepted by the constructor
            for matplotlib.patches.Patch. See
            http://matplotlib.org/api/patches_api.html#matplotlib.patches.Patch
            for a list.
    """
    if keywords is None:
        keywords = {}
    for name, shape in shapes.iteritems():
        object_kwargs = kwargs.copy()
        object_kwargs.update(keywords.get(name, {}))
        object_kwargs['label'] = name

        draw_polygon(axes, shape, **object_kwargs)


def draw_configuration(axes, bodies, configuration, keywords=None, **kwargs):
    """Draw Box2D geometry with the specified configuration

    Args:
        axes: matplotlib axes in which to draw geometry
        bodies (dict): maps object names to box2d bodies
        configuration (dict): maps object names to poses
        keywords (dict): maps object names to **able dicts of keywords
            to be passed to draw_polygon for that object
        kwargs: keyword args to be passed to draw_polygon for each
            object. Valid keywords are those accepted by the constructor
            for matplotlib.patches.Patch. See
            http://matplotlib.org/api/patches_api.html#matplotlib.patches.Patch
            for a list.
    """
    if keywords is None:
        keywords = {}
    for name in configuration:
        shape = metis.geometry.shapely_from_box2d_body(
            bodies[name], configuration[name])

        object_kwargs = kwargs.copy()
        object_kwargs.update(keywords.get(name, {}))
        object_kwargs['label'] = name

        draw_polygon(axes, shape, **object_kwargs)

def draw_undirected_r2graph(axes, graph, **kwargs):
    """Draw a graph whose vertices are points in R2

    Args:
        axes: matplotlib axes in which to draw geometry
        graph: any object which implements the undirected graph
            interface whose vertex data are points in R2
        kwargs: any keyword args accepted by the constructor for
            matplotlib.lines.Line2D for Point, LineString, or LinearRing
            geometries. See
            http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D
            for a list.
    """
    marker_kwargs = kwargs.copy()
    marker_kwargs.setdefault('marker', 'o')
    marker_kwargs.setdefault('markerfacecolor', 'b')
    edge_kwargs = kwargs.copy()
    edge_kwargs.setdefault('marker', '')
    edge_kwargs.setdefault('markerfacecolor', 'b')
    edge_kwargs.setdefault('color', 'k')
    fake_edge_kwargs = edge_kwargs.copy()
    fake_edge_kwargs.setdefault('linestyle', 'dashed')

    seen = set()
    for vertex in graph:
        axes.plot((graph[vertex][0],), (graph[vertex][1],), **marker_kwargs)
        for neighbor in graph.neighbors(vertex):
            if (vertex, neighbor) in seen:
                continue
            else:
                seen.add((vertex, neighbor))
                seen.add((neighbor, vertex))

            if graph.cost(vertex, neighbor) < float('inf'):
                axes.plot((graph[vertex][0], graph[neighbor][0]),
                          (graph[vertex][1], graph[neighbor][1]),
                          **edge_kwargs)
            else:
                axes.plot((graph[vertex][0], graph[neighbor][0]),
                          (graph[vertex][1], graph[neighbor][1]),
                          **fake_edge_kwargs)

def draw_directed_r2graph(axes, graph, **kwargs):
    """Draw a graph whose vertices are points in R2

    Args:
        axes: matplotlib axes in which to draw geometry
        graph: any object which implements the directed graph interface
            and whose vertex data are points in R2
        kwargs: any keyword args accepted by the constructor for
            matplotlib.lines.Line2D for Point, LineString, or LinearRing
            geometries. See
            http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D
            for a list.

    Returns:
        shapely.geometry.Polygon: an axis-aligned rectangle which
            includes all of the vertices
    """
    for vertex in graph:
        for child in graph.successors(vertex):
            if graph.cost(vertex, neighbor) < float('inf'):
                # TODO: figure out how to draw arrows
                axes.plot((graph[vertex][0], graph[child][0]),
                          (graph[vertex][1], graph[child][1]), **kwargs)

