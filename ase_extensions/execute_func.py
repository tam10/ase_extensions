__author__ = 'clyde'

import dill
import sys
from optparse import OptionParser

import ase
import gausspy

def _run(func_fn):
    """Executes dill-ed function/the calculator of an ASE Atoms object using args/kwargs from dill-ed args/kwargs files"""

    session_fn = func_fn.replace('.pkl', '_session.pkl')

    try:
        dill.load_session(session_fn)
    except IOError:
        pass

    with open(func_fn) as func_f:
        func, args, kwargs = dill.load(func_f)

    if func.__class__ == gausspy.gaussian.Gaussian:
        func.start(*args, **kwargs)
    elif func.__class__ == ase.atoms.Atoms:
        func.calc.start(*args, **kwargs)
    else:
        func(*args, **kwargs)

def execute():
    """Exit point allowing running of jobs/functions from dill-ed objects"""

    p = OptionParser(
        usage="usage: %prog my_dill_obj.pkl",
        description="Execute dill-ed objects containing calculations.")

    opts, args = p.parse_args()

    if len(args) != 1:
        p.error('Requires 1 argument: the name of the dilld file to execute')

    fn = args[0]

    _run(fn)

if __name__ == '__main__':
    execute()
