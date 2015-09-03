#!/usr/bin/env python

from setuptools import setup
setup(name='ase_extensions',
      version='0.1',
      packages=['ase_extensions'],

      entry_points = {
        'console_scripts': [
            'execute_calc = ase_extensions.execute_func:execute',
        ],
        }
      )
