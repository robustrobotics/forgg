"""The metis planning library"""
import metis.queue
import metis.hashdict
import metis.iterators
import metis.geometry
import metis.abstract_graphs
import metis.random_geometric_graphs
import metis.dynamics
import metis.factored_random_geometric_graphs

def check_environment():
    """Check environment variables to enable or disable graphical debug

    Check has two phases: first, it checks for the `DISPLAY` variable,
    used in most Linux distributions to indicate the presence of a
    graphical disoplay. If it is unset or empty, graphical debugging is
    disabled.

    Second, it checks for `ENABLE_GRAPHICAL_DEBUG`, a special
    environment variable. If that variable is set to a truthy value (one
    of {1, true, on}), graphical debugging will be enabled; if
    it is set to a falsey value (one of {0, false, off, ""})
    graphical debugging is disabled. If the value is unset, by default
    debugging is enabled; if the value is set to an unknown string, an
    error is raised.

    Returns:
        bool: True if graphical debugging should be enabled

    Raises:
        ValueError if ENABLE_GRAPHICAL_DEBUG is set to an unknown value
    """
    import os
    default_value = True
    if not os.environ.get("DISPLAY"):
        return False
    else:
        env_str = os.environ.get("ENABLE_GRAPHICAL_DEBUG")
        truthy = {"1", "true", "on"}
        falsey = {"0", "false", "off", ""}
        if env_str is None:
            return default_value
        elif env_str.lower() in truthy:
            return True
        elif env_str.lower() in falsey:
            return False
        else:
            raise ValueError(
                "Environment variable 'ENABLE_GRAPHICAL_DEBUG' has invalid "
                "value '{}', Valid true values are {}; false values are {}. "
                "Values are case-insenstive.".format(env_str, truthy, falsey))

"""Flag to determine if graphical debug plots should be displayed

This will be set to a reasonable value automatically (True if a display
is available) and can be manipulated using the environment variable
`ENABLE_GRAPHICAL_DEBUG`. Alternatively, it can be manually overridden
in another script or module; because of how python treats globals, any
checks after it is overridden will be consistent across modules.
"""
GRAPHICAL_DEBUGGING_ENABLED = check_environment()

def graphical_debugging_enabled():
    """Determine if graphical debugging plots should be displayed

    This function is just a thin wrapper around a global variable,
    because I hate to see variables in ALL_CAPS in code.

    Returns:
        bool: True if graphical debugging plots should be displayed
    """
    return GRAPHICAL_DEBUGGING_ENABLED

