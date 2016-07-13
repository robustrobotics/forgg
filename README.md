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
- (optional) matplotlib
- (optional) descartes
- (optional) nosetests


Everything can be installed through pip.
```
pip install -r requirements.txt
```
Several `apt` packages are required for `pip` to install these packages without
errors; the following string of commands will install the dependencies of
this package in a virtualenv.
```
sudo apt-get install -y python-dev python-virtualenv libfreetype6-dev libpng-dev swig libgeos-dev pkg-config
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

If you invoke pip while not in a `virtualenv`, it will install to `/usr/` by
default, and will require `sudo` privileges. If you'd rather install somewhere
else, run
```
pip install --upgrade --install-option="--prefix=$MY_PREFIX" -r requirements.txt
```
with `$MY_PREFIX` set to, for instance, `$HOME/frogg`. Note that you must also
update your `PYTHONPATH` environment variable accordingly to use the packages.

## Usage 

The `frogg` package contains the bookkeeping code required to organize the
random geometric graphs; the planning algorithms themselves are implemented in
the `block_pushing.py` script. To run a job, create a jobfile and call `run.py`.
```
./run.py jobfile
```

