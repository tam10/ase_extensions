__author__ = 'clyde'

import paramiko
import ConfigParser
import os
import time
import uuid


class PBSUtilError(Exception): pass
class PBSUtilQStatError(PBSUtilError): pass
class PBSUtilQSubError(PBSUtilError): pass
class PBSUtilWaitError(PBSUtilError): pass

qstat_c = 'source /home/$USER/.bashrc; qstat '
qdel_c = 'source /home/$USER/.bashrc; qdel '
qsub_c = 'source /home/$USER/.bashrc; qsub '
qmove_c = 'source /home/$USER/.bashrc; qmove '

try:
    config = ConfigParser.RawConfigParser()
    config.read(os.path.expanduser('~/.cc_notebook.ini'))

    user = config.get('pbs', 'user')
    server = config.get('pbs', 'server')

except ConfigParser.NoSectionError:
    config, user, server = None, None, None
    print('\nNo PBS details missing from ~/.cc_notebook.ini. Remote calculations disabled')

def connect_server(usrname=user, servname=server, ssh=True, sftp=False):
    ssh_serv = paramiko.SSHClient()
    ssh_serv.load_system_host_keys()
    ssh_serv.connect(hostname=servname, username=usrname)
    if not sftp:
        return ssh_serv

    sftp_serv = ssh_serv.open_sftp()

    if not ssh:
        ssh_serv.close()
        return sftp_serv
    else:
        return ssh_serv, sftp_serv

class JobStatus:

    def __init__(self, id, state, name=None, elapsed_time=None,
                 username=None, queue=None, nodes=None):
        self.id = id
        self.state = state
        self.name = name
        self.elapsed_time = elapsed_time
        self.username = username
        self.queue = queue
        self.nodes = nodes

    def __str__(self):
        return '%10s %20s         %s   %s' % (self.id, self.name, self.state, self.elapsed_time)

class QueueStatus:

    def __init__(self, id, running, queued, memory=None, time=None, 
                 wall_time=None, node=None, lm=None, state=None):
        self.id = id
        self.running = running
        self.queued = queued
        self.memory = memory
        self.time = time
        self.wall_time = wall_time
        self.node = node
        self.lm = lm
        self.state = state

    def __str__(self):
        return '%10s  R: %i  Q: %i  S: %s' % (self.id, self.running, self.queued, self.state)

def call_qstat(args):
    """Execute qstat, and return output lines"""
    ssh = connect_server()
    i,o,e = ssh.exec_command(qstat_c + ' '.join(args))
    qstat_output = "".join(o.readlines() + e.readlines())
    ssh.close()
    return qstat_output.splitlines()


def parse_qstat_plain_output(output_lines):
    """Parse the output of qstat in the form with no arguments."""

    if len(output_lines) < 3:
        raise PBSUtilQStatError('Bad qstat output:\n"%s"' % '\n'.join(output_lines))

    job_statuses = []

    for output_line in output_lines[2:]:
        job_record = output_line.split()
        record_job_id = parse_qsub_output(job_record[0])[0]
        record_job_state = job_record[4]
        name = job_record[1]
        queue = job_record[2]
        nodes = job_record[6]
        elapsed_time = job_record[3]
        job_statuses.append(JobStatus(record_job_id, record_job_state, name=name, elapsed_time=elapsed_time, queue=queue, nodes=nodes))

    return job_statuses


def parse_qstat_all_output(output_lines):
    """Parse the output of qstat in the form with the -a argument."""

    if len(output_lines) < 1:
        return []

    if len(output_lines) < 3:
        raise PBSUtilQStatError('Bad qstat output:\n"%s"' % '\n'.join(output_lines))

    job_statuses = []

    for output_line in output_lines[5:]:
        job_record = output_line.split()
        record_job_id = parse_qsub_output(job_record[0])[0]
        record_job_state = job_record[9]
        name = job_record[3]
        queue = job_record[2]
        nodes = job_record[6]
        elapsed_time = job_record[10]
        username = job_record[1]
        job_statuses.append(JobStatus(record_job_id, record_job_state, name=name, elapsed_time=elapsed_time,
                                      username=username, queue=queue, nodes=nodes))

    return job_statuses


def qstat_plain():
    """Return a JobStatus object output by qstat for empty argument line."""
    output_lines = call_qstat(['-a'])

    job_statuses = parse_qstat_all_output(output_lines)

    return job_statuses


def qstat_id(job_id):
    """Return a JobStatus object output by a qstat ### request.

    The output for qstat is very different depending on the query, so
    the different queries have been broken into distinct
    functions.

    """

    output_lines = call_qstat([str(job_id)])
    if len(output_lines) != 3:
        raise PBSUtilQStatError('Bad qstat id output:\n"%s"' % '\n'.join(output_lines))

    job_statuses = parse_qstat_plain_output(output_lines)

    assert len(job_statuses) == 1, "qstat id did not return the expected number of job statuses: %s != 1" % len(job_statuses)

    job_stat = job_statuses[0]
    assert job_stat.id == job_id, "qstat job_id did no match expected job_id.  %s != %s" % (job_stat.id, job_id)

    return job_stat

def qstat_user(user):
    """Return a JobStatus object output by a qstat -u user request..

    The output for qstat is very different depending on the query, so
    the different queries have been broken into distinct
    functions.

    """

    job_stats = []

    output_lines = call_qstat(['-u', user])
    if len(output_lines) < 4:
        return job_stats               # No jobs for the current user
    for line in output_lines[5:]:
        job_record = line.split()
        record_job_id = parse_qsub_output(job_record[0])[0]
        record_job_state = job_record[9]
        job_stats.append(JobStatus(record_job_id, record_job_state, name=job_record[3], elapsed_time=job_record[10]))
    return job_stats

def qstat_queue(queue):
    """Return JobStatus object output by a qstat -q queue requrest..

    The output for qstat is very different depending on the query, so
    the different queries have been broken into distinct
    functions.

    """

    job_stats = []

    output_lines = call_qstat(['-q', queue])
    if len(output_lines) < 4:
        return job_stats  

    line = output_lines[5] 
    q,m,ct,wt,n,r,qd,lm,s1,s2 = line.split()
    
    s=' '.join([s1,s2])

    int_values = [m,r,qd]
    for i,v in enumerate(int_values):
        try:
            int_values[i] = int(v)
        except ValueError:
            int_values[i] = None
     
    return QueueStatus(id=q, running=int_values[1], queued=int_values[2], memory=int_values[0], 
                       time=ct, wall_time=wt, node=n, lm=lm, state=s)
    

def qstat(job_id=None, user=None):
    """Return JobStatus objects from output of qstat with desired options."""

    if job_id:
        return [qstat_id(job_id)]
    elif user:
        return qstat_user(user)
    else:
        return qstat_plain()


def parse_qsub_output(output):
    """Divide qsub output into a tuple of job_id and hostname signature."""
    try:
        job_id = output.split('.')[0]
        signature = '.'.join(output[:-1].split('.')[1:]) # the [:-1] kills the newline at the end of the qsub output
        return (job_id, signature)
    except Exception:
        raise PBSUtilQSubError('Unable to parse qsub output: "%s"' % output)


def qsub(script_filename, verbose=False, extra_files=None, extra_commands=''):
    """Submit the given pbs script, returning the jobid."""

    if not extra_files:
        extra_files = []

    local_home = os.path.realpath(config.get('ase', 'ase_home'))
    serv_home = config.get('gaussian', 'gauss_home')

    script_filename = os.path.abspath(script_filename)
    try:
        r_script_loc = script_filename.split(local_home)[1]
        remote_path = os.path.dirname(r_script_loc)
    except IndexError:
        raise RuntimeError('Not running from within ase_home directory')

    ssh, sftp = connect_server(ssh=True,sftp=True)
 
    #make sure server path exists on the server
    i,o,e = ssh.exec_command('mkdir -p {pth};'.format(pth=serv_home + remote_path))
    o.readlines() + e.readlines()
    
    sftp.put(script_filename, serv_home + '/' + r_script_loc)

    #extra files we are also copying into the same directory
    for file_n in extra_files:
        sftp.put(file_n, serv_home + remote_path + '/' + file_n)

    sftp.close()
    i,o,e = ssh.exec_command('cd {pth};'.format(pth=serv_home + remote_path) + qsub_c + ' ' + serv_home + r_script_loc)
    qsub_output = o.readlines() + e.readlines()
    ssh.close()

    if len(qsub_output) == 0:
        raise PBSUtilQSubError("Failed to submit %s, qsub gave no stdoutput.'" % (script_filename))
    if verbose:
        print '\n%s\n' %  qsub_output
    pbs_id = parse_qsub_output(qsub_output[0])[0]
    return pbs_id

def qwait(job_id,sleep_interval=5,max_wait=None):
    try:
        while qstat_id(job_id).state == 'Q':
            time.sleep(sleep_interval)
    except PBSUtilError:
        return

    if not max_wait is None:
        start_time = time.time()
    while True:
        if (not max_wait is None) and time.time() - start_time > max_wait:
            raise PBSUtilWaitError("PBS script failed to return within max_wait time. max_wait=%s" % max_wait)
        try:
            qstat_id(job_id)       # This will throw an exception when the job completes.
            time.sleep(sleep_interval)
        except PBSUtilError:
            break

def qdel(job_id):
    """Kill the given pbs jobid."""
    ssh = connect_server()
    if isinstance(job_id, JobStatus):
        i,o,e = ssh.exec_command(qdel_c + ' ' + job_id.id)
    else:
        i,o,e = ssh.exec_command(qdel_c + ' ' + job_id)

    qdel_output = o.readlines() + e.readlines()
    ssh.close()

def qmove(queue, job_id):
    """Moves the given pbs jobid to the given queue."""
    ssh = connect_server()
    if isinstance(job_id, JobStatus):
        i,o,e = ssh.exec_command(qmove_c + queue + ' ' + job_id.id)
    else:
        i,o,e = ssh.exec_command(qmove_c + queue + ' ' + job_id)

    qmove_output = o.readlines() + e.readlines()
    ssh.close()

def temp_file_name(suffix):
    """Return a new temporary file name."""
    return 'tmp%s%s' % (uuid.uuid4(), suffix)

def get_signature():
    dummy_script_name = temp_file_name('dummy_script')
    open(dummy_script_name, 'w')
    try:
        ssh = connect_server()
        i,o,e = ssh.exec_command(qsub_c + ' ' + dummy_script_name)
        qsub_output = o.readlines() + e.readlines()
        (job_id, signature) = parse_qsub_output(qsub_output)
        ssh.close()
        qdel(job_id)
    finally:
        os.remove(dummy_script_name)

    signature = '.'.join(signature.split('.')[:1])
    return signature

def generic_script(contents,
                   job_name=None,
                   stdout='/dev/null',
                   stderr='/dev/null',
                   shebang='#!/bin/bash',
                   numnodes=None,
                   numcpu=None,
                   queue=None,
                   walltime=None,
                   mem=None,
                   pmem=None):
    """Create a generic pbs script executing contents."""
    me = __file__
    current_time = time.strftime('%H:%M %D')

    if job_name is None:
        job_name = 'unnamed_job'

    if numnodes is None:
        numnodes = str(config.getint('pbs','numnodes'))#str(configuration.numnodes)

    if numcpu is None:
        numcpu = str(config.getint('pbs','numprocs'))#str(configuration.numprocs)

    if pmem:
        pmem = ',pmem=' + pmem
    else:
        pmem=''


    if mem:
        mem = ',mem=' + mem
    else:
        mem=''


    if queue is None:
        queue = config.read('pbs','queue')#configuration.queue

    additional_configuration_lines = []

    if queue is not None:
        additional_configuration_lines.append("#PBS -q %(queue)s" % locals())

    if walltime is None:
        walltime = config.getint('pbs','walltime')#configuration.walltime

    if walltime is not None:
        additional_configuration_lines.append("#PBS -l walltime=%(walltime)s" % locals())

    additional_configuration =  '\n'.join(additional_configuration_lines)

    the_script = """#!/bin/bash
# Created by pbs.py
#PBS -N unnamed_job
#PBS -l npcus=1
#PBS -l walltime=1:00:00
#PBS -j oe

{ad_confg}

{c}
""".format(ad_confg = additional_configuration, c=contents)

    return the_script

def strip_pbs_ids(source):
    """Return a list of the pbs job ids contained in source."""
    signature = get_signature()
    return [qsub_output.split('.')[0] for qsub_output in source.splitlines() if qsub_output.find(signature) > 0]


def get_queue(queue_limits):
    """
    Custom queue generator to avoid overloading the queue. 

    Yields a queue where the number of jobs currently queued on the cluster is less
    than the limits specified in queue_limits, if no such queue is available sleeps
    until one is.
    
    Accepts a dictionary queue_limits. 
    Keys represent the queues to use and values the maximum number of jobs allowed 
    to be queued at any one time.

    An empty string key represents the default queue, all jobs running that are not 
    running on the queues specified in queue_limits will assumed to be running on
    the default queue.
    """

    queues, limits = queue_limits.items()
    queues.pop('')

    while(True):    
        
        queued_jobs = qstat_plain()
        jobs = {queue : [j for j in queued_jobs if j.queue == queue] for queue in queues}        
        jobs[''] = [j for j in queued_jobs if j.queue not in queues]

        for queue in queues:
            if len(jobs[queue]) < queue_limits[queue]:
                yield queue
        else:
            time.sleep(30)
