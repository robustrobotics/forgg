# metis

This package is contains tools for evaluating models and inference algorithms
for task and motion planning. The name comes from Metis, a Greek goddess of
wisdom, mother of Athena.

## Maintainer
- Will Vega-Brown (wrvb@csail.mit.edu)

## Dependencies
- numpy
- scipy
- shapely
- pybox2d
- triangle
- (optional) pydot
- (optional) nosetests

Everything can be installed through pip.
```
pip install numpy scipy shapely Box2D triangle
```
If you are not in a `virtualenv`, this will install to `/usr/` by default, and
will require `sudo` privileges. If you'd rather install somewhere else, run
```
pip install --upgrade --install-option="--prefix=$MY_PREFIX" numpy scipy shapely Box2D triangle
```
with `$MY_PREFIX` set to, for instance, `$HOME/rrg`. Note that you must also
update your `PYTHONPATH` environment variable accordingly to use the packages.

NOTE: installing the `pybox2d` from `pip` might fail if `swig` is not installed.
If that happens, `apt-get install swig` may help.

## Installation

This will provide minimal output. To run the tests and see the normal `nose`
output, from the package directory run `nosetests` without arguments.

## Usage 

A simple block-pushing example is included; more comprehensive and extensible
examples are planned for short term development in the future. This example
world has a robot pushing objects between three rooms. To run the example, you
must create a yaml description of the problem to solve. Examples of the format
can be found in the `cfg` directory. Briefly, a valid job file specifies the
start configuration of the world, as well as algorithmic parameters. For
example:
```yaml
noupper_blocked_forgg_0200_00: # name for this job
  algorithm: forgg             # the algorithm to use ('forgg' or 'tamp')
  domain:                      # Parameters affecting the problem domain
    upper_door: True           # True if there should be two doors between the
                               # rooms; False if there should be only one.
  instance:                    # Parameters affecting the problem instance
    box1:                      # The initial location of the first object
    - 3                        #  x coordinate (in meters)
    - 5                        #  y coordinate (in meters)
    - -0.2                     #  orientation (in radians)
    box2:                      # The initial location of the second object
    - 5
    - 2.5
    - 0.1
    robot:                     # The initial location of the robot
    - 2
    - 2
    - 0
  solver:                      # Solver parameters
    count: 200                 # Number of samples to generate in each
                               # mode factor
    seed: 0                    # PRNG seed value 
```

To run this example, call 
```
./run.py cfg/forgg.yaml
```
Command line output is sent to a log file by default, which you can view in real
time with 
```
tail -f noupper_blocked_forgg_0200_00.out
```
To solve the example problem with `count=200`, the planner must explre around
240000 vertices. On a typical desktop, the planner can explore around 200-400
vertices per second, for a total planning time of 10-20 minutes.

