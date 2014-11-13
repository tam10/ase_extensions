__author__ = 'clyde'

import pickle
import sys
import ase


def _run(func_fn):
    """Executes pickled function/the calculator of an ASE Atoms object using args/kwargs from pickled args/kwargs files"""
    with open(func_fn) as func_f:
        func, args, kwargs = pickle.load(func_f)

    if func.__class__ == ase.calculators.gaussian.Gaussian:
        func.start(*args, **kwargs)
    elif func.__class__ == ase.atoms.Atoms:
        func.calc.start(*args, **kwargs)
    else:
        func(*args, **kwargs)

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) > 1:
        raise RuntimeError("Invalid number of arguments")

    _run(args[0])