#!/usr/bin/env python

#qsub -l nodes1,mem=200mb -v obj_file='test' ase_calcs.py

#PBS -v PYTHONPATH
#PBS -j oe

home = '/home/cjf05/'
work = '/work/cjf05'

import os
inp_f = os.environ['inp_f']
host_d = os.environ['host_d']

inp_f = home + host_d + inp_f
out_f = work + host_d + inp_f.replace('.com', '.log')

os.system('module load gaussian; g09 <{i}> {o}'.format(i=inp_f, o=out_f))

#import pickle
#import os

#pickled_obj_file = os.environ["obj_file"]

#calc_obj = pickle.load(pickled_obj_file)
#calc_obj.start()

import pickle
def run_on_cx1(object, job_details):
    with open('temp', 'w') as temp_f:
        pickle.dump(object, temp_f)

    copy_to_cx1('temp')
    ssh.run(cx1_execution_script('temp', job_details))


import os
def cx1_execution_script(file_n, job_details):
    os.system('')