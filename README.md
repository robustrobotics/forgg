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
This code is intended to be used as a ROS package, like most RRG code, and can
be installed using the normal catkin commands.
```
catkin build --this
```

To use this code without ROS or catkin, install it as a normal python package:
```
python setup.py install --prefix=$MYPREFIX
```
Omitting the prefix will install to the system path and will require `sudo`.

## Usage 

There are currently no examples to run, as the package is incomplete. To run the
tests from anywhere in the `rrg` repository, use the normal catkin command:
```
catkin build metis -v --make-args test
```

This will provide minimal output. To run the tests and see the normal `nose`
output, from the package directory run `nosetests` without arguments.

### Recording video
I use ffmpeg to record videos. The version in the ubuntu repositories is a fork
(`avconv`) which I couldn't get to work, but the 'real' ffmpeg works fine. You
can get it from the ppa `ppa:mc3man/trusty-media`, by following these steps.
```
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt-get upgrade
```
If ffmpeg isn't found or fails, the code will let you know, but shouldn't crash.

