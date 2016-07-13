#!/usr/bin/env python
# coding: utf-8
"""Run experiments"""

import time
import multiprocessing
import logging
import argparse

import yaml
import block_pushing

def print_nested(nested_dict, printer, indent=0):
    """Pretty-print a nested dictionary"""
    for key, value in nested_dict.iteritems():
        printer('  ' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_nested(value, printer, indent+1)
        else:
            printer('  ' * (indent+1) + str(value))

class Callback(object):
    """Callback to announce when a job is done"""
    #pylint: disable=too-few-public-methods
    def __init__(self, total):
        super(Callback, self).__init__()
        self.total = total
        self.start = time.time()
        self.counter = 0

    def __call__(self, result):
        success, name = result
        self.counter += 1
        print "[ {} / {} ] {} completed {} in {} s".format(
            self.counter, self.total, name,
            'successfully' if success else 'with errors',
            time.time()-self.start)

def run(task_name, task_parameters):
    """Set up log and run a job"""
    log = logging.getLogger(task_name)
    formatter = logging.Formatter('%(asctime)s:   %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S')
    handler = logging.FileHandler(task_name + ".out", mode='w')
    handler.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.info("Running task {}. Parameters:".format(task_name))
    print_nested(task_parameters, log.info)
    # An exception in a worker (say, due to out of memory errors) can cause
    # multiprocessing to hang forever on join. Instead, catch and log
    # the exception.
    try:
        success = block_pushing.run(task_name, log, task_parameters)
        return success, task_name
    except Exception: # pylint: disable=broad-except
        log.exception("Error in task. Terminating.")
        return False, task_name

def main():
    """Run experiments"""
    parser = argparse.ArgumentParser(
        description="Run experiments for WAFR 2016 paper")
    parser.add_argument('jobs', help='job file to run')
    args = parser.parse_args()

    with open(args.jobs, 'r') as job_file:
        jobs = yaml.load(job_file)

    print "Starting {} jobs".format(len(jobs))
    callback = Callback(len(jobs))
    # Setting maxtasksperchild forces the child process to restart
    # after each task. This will force C libraries like numpy to
    # clean up, which they iothewise wouldn't do. That isn't a major
    # problem for me here, but the fix doesn't hurt anything either.
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1,
                                maxtasksperchild=1)

    for name, job in jobs.iteritems():
        pool.apply_async(run, (name, job,), callback=callback)
    pool.close()
    pool.join()
    print "Done"

if __name__ == "__main__":
    main()
