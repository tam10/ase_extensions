#! /usr/bin/env python


import numpy as np
import re
import os
import molmod
import pickle
import ConfigParser
import remote
from copy import deepcopy
from .shared import plaintext2html

config = ConfigParser.RawConfigParser()
config.read(os.path.expanduser('~/.cc_notebook.ini'))

#import pandas
#import pandas.core.format as fmt
##not yet developed (would be nice to have a dataframe with a jmol panel next to it and each row associated with a molecular structure
#class DataFrame(pandas.DataFrame):
#    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
#        super(DataFrame, self).__init__(data, index, columns, dtype, copy)
#
#    def _repr_html_(self):
#       """
#       Return a html representation for a particular DataFrame.
#       Mainly for IPython notebook.
#       """
#       if fmt.print_config.notebook_repr_html:
#           if self._need_info_repr_():
#               return None
#           else:
#               return ('<div>\n' + self.to_html() + '\n</div>')
#       else:
#           return None

def view_avo(atoms):
    from ase.visualize import view
    if atoms._calc and atoms.calc.__module__ == 'gausspy.gaussian':
        os.system("avogadro {f}".format(f=atoms.calc.label + '.log'))
    else:
        print('Viewing xyz file')
        view(atoms,viewer='avogadro')

def calcs_complete(list_mols):
    for mol in list_mols:
        try:
            if not mol.calc.calc_complete or np.isnan(mol.calc.energy_zero):
                return False
        except AttributeError:
            return False
    return True

def get_incomplete_calcs(list_mols):
    incomp_calcs = []

    for mol in list_mols:
        try:
            if not mol.calc.calc_complete or np.isnan(mol.calc.energy_zero):
                incomp_calcs.append(mol)
        except AttributeError:
            incomp_calcs.append(mol)
    return incomp_calcs


def gen_html_file(text_f):
    html_f = text_f + '.html'
    with open(text_f) as f:
        content = f.read()
        if content:
            html_content = plaintext2html(content)
        else:
            html_content = "<p></p>"

    with open(html_f, 'w') as f:
        f.write(html_content)


#will only work if called on the original calculation, calling on an intermediate restart
#e.g. orig_calc_name_restart_2.log will not return orig_calc_name_restart_5.log even if it is the final calculation
def get_latest_restart_name(file_n, ssh=None):
    """get's the file name of the final restarted calculation from the server"""
    ssh_created = False
    if not ssh:
        ssh = remote.connect_server(ssh=True)
        ssh_created = True

    i,o,e = ssh.exec_command('ls {fn}_restart_{{?,??}}.log'.format(fn=file_n.replace('.log', '')))
    ls_output = o.readlines()
    clean_ls = [fn.strip() for fn in ls_output]
    clean_ls.sort(key=lambda fn: int(re.search('[0-9]+.log', fn).group().replace('.log' , '')))

    if ssh_created:
        ssh.close()

    try:
        return clean_ls[-1]
    except IndexError:
        return file_n

def gen_fchk(mol_nm, ssh=False):
    """generates fchk file from chk point file for the molecule specified
    assumes chk point file exists in the scratch directory"""
    ssh_created = False
    if not ssh:
        ssh = remote.connect_server(ssh=True)
        ssh_created = True

    i,o,e = ssh.exec_command('/home/gaussian-devel/gaussiandvh13_pgi_118/gdv/formchk {fn}.chk {fn}.fchk'.format(fn=mol_nm))
    formchk_error = e.readlines()

    if ssh_created:
        ssh.close()

    if formchk_error:
        return False
    else:
        return True



def unsymmetrize(atoms, seed=None):
    import numpy as np

    if seed:
        np.random.seed(seed)

    old_positions = atoms.positions
    shape=old_positions.shape
    jiggle_matrix = np.random.rand(*shape)/1*10**-6

    atoms.old_positions = old_positions
    atoms.positions += jiggle_matrix

def gen_fchks(list_mols):
    """generates fchk files from chk point files for the molecules specified.
    assumes chk point files exist in the scratch directory"""
    #scratch, local_home = os.environ['GAUSS_SCRATCH'], os.path.realpath(os.environ['ASE_HOME'])
    scratch, local_home = config.get('gaussian', 'gauss_scratch'), os.path.realpath(config.get('ase', 'ase_home'))

    try:
        active_dir = os.getcwd().split(local_home)[1]
        scratch_dir = scratch + active_dir
    except IndexError:
        raise RuntimeError('Not running from within ASE_HOME')

    home_files = [mol.calc.label for mol in list_mols]
    serv_files = [scratch_dir + '/' + fn for fn in home_files]

    ssh = remote.connect_server(ssh=True)
    fchk_out = [gen_fchk(serv_f, ssh) for serv_f in serv_files]
    ssh.close()

    return fchk_out


def get_active_dirs():
    #scratch, home, local_home = os.environ['GAUSS_SCRATCH'], os.environ['GAUSS_HOME'], os.path.realpath(os.environ['ASE_HOME'])
    scratch = config.get('gaussian', 'gauss_scratch')
    home = config.get('gaussian', 'gauss_home')
    local_home = os.path.realpath(config.get('ase', 'ase_home'))

    try:
        active_dir = os.getcwd().split(local_home)[1]
        home_dir = home + active_dir
        scratch_dir = scratch + active_dir
    except IndexError:
        raise RuntimeError('Not running from within ASE_HOME')

    return home_dir, scratch_dir


def check_calcs_v2(list_mols, data_file="", max_restart=False, depth='medium', frc=False):
    """if log/fchk file data on server differ from saved data,extract data and save"""
    import warnings
    from gausspy.gaussian_job_manager import server_data_unequal
    from gausspy.data_extract_utils import latest_restarts, import_moldata, load_from_server
    from gausspy.data_extract_utils import oniom_components_on_server, export_moldata, clean_local_files
    from gausspy import oniom_utils

    if max_restart:
        list_mols = latest_restarts(list_mols)

    #if we are forcing we ignore previously saved data
    if not frc:
        current_data = import_moldata(data_file)
    else:
        current_data = []

    #check saved data against current list of molecules
    current_labels = [m.calc.label for m in current_data]
    mol_labels = [m.calc.label for m in list_mols]
    mismatched_labels = [label for label in current_labels if label not in mol_labels]

    if mismatched_labels:
        warnings.warn(
            RuntimeWarning(
                "Calculations: {m} in data_file do not match molecules passed to check_calcs".format(
                    m=" ".join(mismatched_labels))
            )
        )

    #extract calculation data from the datafile (note at the moment we are not extracting the non calculation part
    # which means that if the calculation modifies the original ase object those changes will be lost
    for saved_mol in current_data:
        try:
            ind = mol_labels.index(saved_mol.calc.label)
            #loading the entire object = list_mols[ind] = saved_mod
            #but because of python's reference passing behaviour this would mean check_calcs_v2 would not act like check_calcs (think restarted incomplete calculations)
            list_mols[ind].calc = saved_mol.calc
        except ValueError:
            pass

    if frc:
        update_mask = [True for i in range(len(list_mols))]
    else:
        update_mask =server_data_unequal(list_mols)

    mols_to_update = [list_mols[i] for i in range(len(list_mols)) if update_mask[i]]
    for mol in mols_to_update:
        #if log files on home directory and server are different we copy those files to the home directory
        mol = load_from_server(mol, depth)

        #if we have an oniom calculation we check to see if the components of the calculation have been run and if so we retrieve them and attach them to the calculation object
        if 'oniom' in mol.calc.method and oniom_components_on_server(mol):
            init_mol = copy.deepcopy(mol)
            init_mol.calc.label += '_init'
            init_mol = load_from_server(init_mol, depth)
            mol.calc.components = oniom_utils.oniom_comp_calcs(init_mol)
            mol.calc.components = check_calcs_v2(mol.calc.components, depth=depth, max_restart=True, frc=frc)

    if data_file and any(update_mask):
        export_moldata(data_file, list_mols)
        clean_local_files(mols_to_update)

    return list_mols

def check_calcs(list_mols, max_restart=False, depth='medium', sort=False, frc=False, wait=False):
    """if output file not present copies the output from the server's scratch dir to the working dir, if data_file given data is extracted from output files and saved to the data_file
    the local version of the output files are then deleted"""
    from gausspy import gaussian_job_manager
    from gausspy import oniom_utils
    from gausspy.gaussian_job_manager import server_files_equal_v2
    #scratch, local_home = os.environ['GAUSS_SCRATCH'], os.path.realpath(os.environ['ASE_HOME'])
    scratch = config.get('gaussian', 'gauss_scratch')
    local_home = os.path.realpath(config.get('ase', 'ase_home'))

    try:
        active_dir = os.getcwd().split(local_home)[1]
        scratch_dir = scratch + active_dir
    except IndexError:
        raise RuntimeError('Not running from within ASE_HOME')

    if depth == 'light':
        for mol in list_mols:
            mol.calc.reset_cached_data()
            mol.calc.read()
        return

    home_files = [mol.calc.label + '.log' for mol in list_mols]
    serv_files = [scratch_dir + '/' + fn for fn in home_files]

    if max_restart:
        ssh = remote.connect_server(ssh=True)
        serv_files = [get_latest_restart_name(file_n, ssh) for file_n in serv_files]
        home_files = [sfn.replace(scratch_dir + '/', '') for sfn in serv_files]
        ssh.close()

        for i, mol in enumerate(list_mols):
            mol.calc.label = home_files[i].replace('.log','')

    if frc:
        files_equal = [False for i in range(len(list_mols))]
    else:
        files_equal = server_files_equal_v2(serv_files, home_files)

    if depth == 'heavy' and frc:
        fchk_files_equal = [False for i in range(len(list_mols))]
    elif depth == 'heavy':
        fchk_home_files = [f.replace('.log', '.fchk') for f in home_files]
        fchk_serv_files = [scratch_dir + '/' + fn for fn in fchk_home_files]
        fchk_files_equal = server_files_equal_v2(fchk_serv_files, fchk_home_files)
    else:
        fchk_files_equal = [True for i in range(len(list_mols))]

    #sanity check
    if len(files_equal) != len(list_mols):
        raise RuntimeError('Problem with file checking')

    for i, mol in enumerate(list_mols):
        mol.calc.reset_cached_data()

        #if log files on home directory and server are different we copy those files to the home directory
        if not files_equal[i]:
            mol.calc.get_from_scratch(mol.calc.label + '.log', frc=frc)
        if not fchk_files_equal[i]:
            mol.calc.get_from_scratch(mol.calc.label + '.fchk', frc=frc)
        #if we have an oniom calculation we check to see if the components of the calculation have been run and if so we retrieve them and attach them to the calculation object

        if 'oniom' in mol.calc.method and gaussian_job_manager.server_file_exists(scratch_dir + '/' + mol.calc.label + '_init.log'):
            init_mol = copy.deepcopy(mol)
            init_mol.calc.label += '_init'
            init_mol.calc.get_from_scratch(init_mol.calc.label + '.log', frc=frc)
            mol.calc.components = oniom_utils.oniom_comp_calcs(init_mol)
            check_calcs(mol.calc.components, depth=depth, max_restart=True, frc=frc)

    for mol in list_mols:
        try:
            mol.calc.read()
        except (AttributeError, KeyError):
            pass

    if calcs_complete(list_mols):
        return "Calculations complete"

    incomp_calcs = get_incomplete_calcs(list_mols)

    if sort:
        #sort by calculation status
        incomp_calcs.sort(key=lambda e: e.calc.status)

    if wait:
        while([m for m in list_mols if m.calc.active]):
            time.sleep(15)

        check_calcs(list_mols=list_mols, max_restart=max_restart, depth=depth, sort=sort, frc=frc, wait=wait)

    return incomp_calcs

def move_calc_files(ase_obj, new_label):
    """ Change the name of a gaussian calculation, changes com,log,chk,fchk file names to the label given"""
    old_label = ase_obj.calc.label
    home, scratch = get_active_dirs()

    home_exts, scratch_exts = ['.com'], ['.log', '.chk', '.fchk']

    ssh = remote.connect_server(ssh=True)

    commands = []
    for ext in home_exts:
        command = ''.join(['mv', home, old_label, ext, ',', home, new_label, ext])
        commands.append(command)
    for ext in scratch_exts:
        command = ''.join(['mv', scratch, old_label, ext, ',', scratch, new_label, ext])
        commands.append(command)

    for command in commands:
        i,o,e = ssh.exec_command(command)

    ssh.close()

import time
#todo
def wait_until_complete(ase_obj):
    check_calcs([ase_obj])
    if not ase_obj.calc.calc_complete:
        time.sleep(60)

    return


def get_oniom_stable_calcs(mol):
    from gausspy import oniom_utils
    comps = [copy.deepcopy(mol) for i in range(3)]
    comp_strs = ['low_real', 'high_model', 'low_model']
    methods = oniom_utils.get_oniom_calc_methods(mol.calc.method)
    for i,m in enumerate(comps):
        m.calc.method = methods[i]
        m.calc.label += '_init_' + comp_strs[i]

    return comps

def get_equiv_scratch_dir():
    #scratch = os.environ['GAUSS_SCRATCH']
    #local_home = os.path.realpath(os.environ['ASE_HOME'])

    scratch = config.get('gaussian', 'gauss_scratch')
    local_home = os.path.realpath(config.get('ase', 'ase_home'))

    try:
        active_dir = os.getcwd().split(local_home)[1]
    except IndexError:
        raise RuntimeError('Not running from within ASE_HOME')

    scratch_dir = scratch + active_dir +'/'
    return scratch_dir

def get_equiv_home_dir():
    #home = os.environ['GAUSS_HOME']
    #local_home = os.path.realpath(os.environ['ASE_HOME'])

    home = config.get('gaussian', 'gauss_home')
    local_home = os.path.realpath(config.get('ase', 'ase_home'))

    try:
        active_dir = os.getcwd().split(local_home)[1]
    except IndexError:
        raise RuntimeError('Not running from within ASE_HOME')

    home_dir = home + active_dir + '/'

    return home_dir

def scratch_cp(file1, file2):
    #serv = os.environ['GAUSS_HOST']
    serv = config.get('gaussian', 'gauss_host')
    scratch_dir = get_equiv_scratch_dir()
    exitcode = os.system("ssh " + serv + " 'cp " + scratch_dir + "{f1} ".format(f1=file1) + scratch_dir + "/" + "{f2}'".format(f2=file2))

    if exitcode == 0:
        return 'Success'
    else:
        return 'Fails with code: {c}'.format(c=exitcode)

def kill_mol(mol):

    #serv = os.environ['GAUSS_HOST']
    serv = config.get('gaussian', 'gauss_host')

    scratch_dir = get_equiv_scratch_dir()
    home_dir = get_equiv_home_dir()
    os.system("rm " + mol.calc.label + '.log')
    os.system("rm " + mol.calc.label + '.com')
    exit_code =  os.system("ssh " + serv + " 'rm " + home_dir + mol.calc.label + ".com; " +
                              "rm " + home_dir + mol.calc.label + "_job.sh; " +
                              "rm " + scratch_dir + mol.calc.label + ".log; " +
                              "rm " + scratch_dir + mol.calc.label + ".chk'")

    if exit_code == 0:
        return 'Success'
    else:
        return 'Failed'

def get_uncut_indices(atoms, orig, vec1, offset=None):
    """a plane is defined and atoms on one side of this plane are cut away,
    orig specifies the origin
    vec1-orig specifies the direction normal to the plane
    offset is an real float specifying the amount to move the origin along vec1"""

    if isinstance(orig, int):
        orig = atoms.get_positions()[orig]
    else:
        orig = np.array(orig)
    if isinstance(vec1, int):
        vec1 = atoms.get_positions()[vec1]
    else:
        vec1 = np.array(vec1)

#    if vec2 and isinstance(vec2, int):
#        vec2 = atoms.get_positions()[vec2]
#    elif vec2:
#        vec2 = np.array(vec2)
#    if vec3 and isinstance(vec3, int):
#        vec3 = atoms.get_positions()[vec3]
#    elif vec3:
#        vec3 = np.array(vec3)
#
#    if vec2:
#        plane_vec = np.cross(vec1-orig, vec2-orig)
#    else:
#        plane_vec = vec1-orig
#
#    nplane_vec = plane_vec/np.linalg.norm(plane_vec)
#
#    if vec2 and vec3 and nplane_vec.dot(vec3-orig) < 0:
#        nplane_vec *= -1

    plane_vec = vec1 - orig
    nplane_vec = plane_vec/np.linalg.norm(plane_vec)

    if offset:
        orig += offset*nplane_vec

    atom_indices = [i for i,a in enumerate(atoms) if (a.position - orig).dot(nplane_vec) > 0]
    return atom_indices

def cut(atoms, orig, vec1, offset=None):
    atom_indices = get_uncut_indices(atoms, orig, vec1, offset)
    return atoms[atom_indices]

def multicut(atoms, origs, vec1s, offsets = None):
    if not offsets:
        offsets = []
    if len(origs) != len(vec1s) != len(offsets):
        raise RuntimeError("lists of vectors and offsets must be the same size")

    cut_indices = []
    atom_indices = range(len(atoms))
    for i in range(len(origs)):
        current_slice_indices=get_uncut_indices(atoms, origs[i], vec1s[i], offsets[i])
        cut_indices += [i for i in atom_indices if i not in current_slice_indices]

    slice_indices = [i for i in atom_indices if i not in cut_indices]
    return atoms[slice_indices]

def join(atoms1, atoms2):
    new_mol = deepcopy(atoms1)
    for a in atoms2:
        new_mol.append(a)
    return new_mol

def get_neighbours(atoms, atom, n=None):
    if not n:
        n= len(atoms)

    temp_atoms = [a for a in atoms if not all(a.position == atom.position)]

    distances = []
    for a in temp_atoms:
        distances.append(np.linalg.norm(atom.position-a.position))

    list_neighbour_dists = zip(temp_atoms, distances)
    list_neighbour_dists.sort(key = lambda e: e[1])
    return zip(*list_neighbour_dists[0:n])[0]

def align(atoms, init_vec, align_vec):
    import transformations
    import numpy as np

    if len(init_vec) == 2:
        orig_atom_ind = init_vec[0]
        final_atom_ind = init_vec[1]
        init_vec = atoms[final_atom_ind].position - atoms[orig_atom_ind].position

    if len(align_vec) == 2:
        orig_atom_ind = align_vec[0]
        final_atom_ind = align_vec[1]
        align_vec = atoms[final_atom_ind].position - atoms[orig_atom_ind].position

    rot_axis = np.cross(init_vec, align_vec)
    nrot_axis = rot_axis/np.linalg.norm(rot_axis)

    angle = transformations.angle_between_vectors(init_vec, align_vec)

    rot_M = transformations.rotation_matrix(angle, nrot_axis)[:3,:3]

    for a in atoms:
        a.position = rot_M.dot(a.position)

    return atoms


def bonded_neighbours(atom, pot_neighbours):
    numbers = pot_neighbours.calc.data['Atomic_numbers']
    coordinates = pot_neighbours.get_positions()
    title = pot_neighbours.calc.label
    symbols = pot_neighbours.get_chemical_symbols()

    mol = molmod.Molecule(numbers, coordinates, title, symbols)
    mol.set_default_graph()
    atom_ind = pot_neighbours.index(atom)
    neighb_inds = mol.graph.neighbours[atom_ind]

    return [n for i, n in enumerate(pot_neighbours) if i in neighb_inds]

import copy
#start is either be a list of the coordinates of the first atom (a list not a numpy array (because of the boolian nature of np array is weird)
#or it is the index of the atom of the first atom (indices treated as python/c i.e. first atom has an index of zero)

def reorder_xyz(xyz_filen, elements = False, start=None, primary_atoms=None, secondary_atoms=None):
    """reorders an xyz file, first by whether atoms are specified as primary or secondary atoms, then by element atomic no.
       then in order of distance to the provided coords, coords can be provided as absolute coordinates or
       by referring to the coordinates of a particular atom"""
    with open(xyz_filen) as xyz_file:
        data = xyz_file.readlines()

    get_coords = lambda e: np.array(map(float,e.split()[1:]))
    dist = lambda e: np.linalg.norm(get_coords(e)-np.array(start_coords))

    start_coords = None
    if start:
        try:
            start_coords = [c for c in start]
        except TypeError:
            start_coords = list(get_coords(data[start+2]))

    header = data[0:2]
    contents = data[2:]

    #if no primary_atoms we are not splitting into primary and secondary
    if primary_atoms and secondary_atoms:
        raise RuntimeError('Redundany selection of both primary and secondary atoms')

    if not primary_atoms and not secondary_atoms:
        primary_atoms = range(len(contents))
    if primary_atoms:
        secondary_atoms = [i for i in range(len(contents)) if i not in primary_atoms]
    if secondary_atoms:
        primary_atoms = [i for i in range(len(contents)) if i not in secondary_atoms]

    primary_contents = copy.deepcopy([line for n,line in enumerate(contents) if n in primary_atoms])
    secondary_contents = copy.deepcopy([line for n,line in enumerate(contents) if n in secondary_atoms])

    if start_coords:
        primary_contents.sort(key=dist)
        secondary_contents.sort(key=dist)

    if elements:
        primary_contents = sort_by_ele(primary_contents)
        secondary_contents = sort_by_ele(secondary_contents)

    new_data = header + primary_contents + secondary_contents

    pth,fl = os.path.split(os.path.abspath(xyz_filen))
    with open(pth + '/sorted_' + fl ,'w') as xyz_file:
        xyz_file.writelines(new_data)

def sort_by_ele(contents):
    from ase.data import atomic_numbers
    """sorts the contents of an xyz file by element in the order provided by the elements list variable"""
    get_char = lambda e: e.split()[0]
    char_signif = lambda e: atomic_numbers[get_char(e)]
    contents.sort(key=char_signif,reverse=True)
    return contents

def to_molmod(ase_atoms):
    import molmod, tempfile, os
    from ase.io import write

    fd, xyz_f = tempfile.mkstemp('.xyz', 'ase-')
    fd = os.fdopen(fd, 'w')
    write(fd, ase_atoms, format='xyz')
    fd.close()

    mol = molmod.Molecule.from_file(xyz_f)
    os.remove(xyz_f)

    mol.set_default_graph()
    mol.set_default_symbols()

    return mol


def from_sdf(mol_fn):
    """Constructs an ase object from a sdf mol file as constructed by gaussview"""
    from ase import Atoms

    with open(mol_fn) as mol_f:
        mol_data = mol_f.readlines()

    coord_section = [l for l in mol_data if len(l.split()) == 16]
    atom_symbols = [l.split()[3] for l in coord_section]
    str_coords = [l.split()[:3] for l in coord_section]
    coords = [map(float, atom_coords) for atom_coords in str_coords]

    return Atoms(symbols=atom_symbols, positions=coords)


#see Geom_convergence_testing notebook for a slightly more involved variant plugging into pymol
def bond_dist_delta(ase_mol1, ase_mol2):
    """bond_dist_delta returns two lists of the same length, the first is a list of bond indices, the second is a list of the changes in the bond length corresponding to that index"""
    #convert to molmod
    mol1 = to_molmod(ase_mol1)
    mol2 = to_molmod(ase_mol2)

    #get bond distances between neighbouring carbon atoms
    mol1_bdists_inds = bond_distances_v2(mol1)
    #seperate the bond distances and the atom indices the bonds correspond to
    #nb indexes are python_like so start at zero programs (e.g. pyMol/Avogadro) often number atoms starting at 1
    mol1_bdists, mol1_inds = zip(*mol1_bdists_inds)

    mol2_bdists_inds = bond_distances_v2(mol2, bonds=mol1_inds)
    mol2_bdists, mol2_inds = zip(*mol2_bdists_inds)

    if mol1_inds != mol2_inds:
        raise RuntimeError('Comparison of bond distances for different molecules not yet implemented')

    mol1_bdists = np.array(mol1_bdists)
    mol2_bdists = np.array(mol2_bdists)

    delta_bdists = mol1_bdists - mol2_bdists
    return np.array([mol1_inds, delta_bdists])

def bond_distances_v2(molmod_atoms, bonds=None, ignored_elements=None):
    """Computes bond distances for a molmod molecule"""
    if not ignored_elements:
        ignored_elements = []

    m=molmod_atoms

    if not bonds:
        bonds = m.graph.edges

    bond_dists = []
    indices = []

    for ind1, ind2 in bonds:
        if not m.symbols[ind1] in ignored_elements and not m.symbols[ind2] in ignored_elements:
            bond_dists.append(m.distance_matrix[ind1,ind2]/molmod.angstrom)
            indices.append((ind1, ind2))

    #we sort by bond index so that comparison between two bdist_inds objects is possible (without sorting we can get variation in the order)
    bdist_inds = zip(bond_dists, indices)
    bdist_inds.sort(key=lambda e: e[1])

    return bdist_inds


def max_dihedrals(atoms):

    def get_dihedral(p):
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array([v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        return np.degrees(np.arctan2(y, x))

    def length_n_frags(mol, initial):
        """Returns all chains of length n that start at carbon initial"""
        frags = []
        current_frag = initial
        if len(current_frag) >= 4:
            return [current_frag]

        neighbor_indices = mol.graph.neighbors[current_frag[-1]]
        for neighbor_ind in neighbor_indices:
            if neighbor_ind not in current_frag:
                new_frag = current_frag + (neighbor_ind, )
                frags += length_n_frags(mol, new_frag)
        return frags

    mol = to_molmod(atoms)

    data = []
    for i in range(len(atoms)):
        chains = length_n_frags(mol, (i,))
        for chain_indices in chains:
            atom_positions = np.array([atoms[temp_index].position for temp_index in chain_indices])
            dihedral = get_dihedral(atom_positions)
            result = (chain_indices, dihedral)
            data.append(result)

    chains, dihedrals = zip(*data)
    abs_dihedral_f = lambda a: abs(a) if abs(a) < 90 else abs(abs(a)-180)
    abs_dihedrals = [abs_dihedral_f(d) for d in dihedrals]
    proc_data = zip(chains,abs_dihedrals)
    proc_data.sort(key=lambda e: e[1])
    return proc_data

def sp2_dihedrals(atoms):
    """Returns a list of indices, dihedrals for an atoms object, dihedrals are reported for atoms with 3 neighbors"""

    #problems with atoms inbuilt dihedral method (doesn't match gaussview/jmol at all)
    #so we'll use one taken from http://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    def get_dihedral(p):
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array([v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        return np.degrees(np.arctan2(y, x))

    mol = to_molmod(atoms)
    data = []

    for i in range(len(atoms)):
        if len(mol.graph.neighbors[i]) == 3:
            atom_indices = [i] + list(mol.graph.neighbors[i])
            atom_positions = np.array([atoms[temp_index].position for temp_index in atom_indices])
            #dihedral = atoms.get_dihedral(atom_indices)
            dihedral = get_dihedral(atom_positions)
            result = (i, dihedral)
            data.append(result)

    return data

#no longer used
#def bond_distances(molmod_atoms, atom_select_fs=None):
#    import numpy as np
#    import molmod
#
#    if not atom_select_fs:
#        atom_select_fs = []
#
#    m = molmod_atoms
#    #gets mask for atoms one bond away
#    mask = m.graph.distances==1
#    #to avoid double counting we get rid of the top triangle
#    mask &= np.tri(len(m.graph.distances))==1
#
#    for fn in atom_select_fs:
#        fn_mask = fn(m)
#        mask = mask & fn_mask
#
#    #need to invert the mask to use a masked array
#    mask2 = mask == False
#
#    data=np.ma.MaskedArray(m.distance_matrix, mask2)
#
#    bond_dists = np.array([datum/molmod.angstrom for row in data for datum in row if datum])
#    indices = zip(*data.nonzero())
#    return zip(bond_dists, indices)
#
##example atom_select_fn
#def get_only_element_mask(molmod_atoms, ele_symbol='C'):
#    import numpy as np
#    m=molmod_atoms
#
#    mask = np.zeros_like(m.distance_matrix)
#    size = len(m.distance_matrix)
#    for i in range(size):
#        for j in range(size):
#            if m.symbols[i] == ele_symbol and m.symbols[j] == ele_symbol:
#                mask[i,j] = 1
#    return mask == 1

#useful when moving from data indexed in Python/C style to data indexed in Fortran style
def increment_all(lst):
    if not lst:
        return []
    else:
        new_lst = [e+1 for e in lst]
        return new_lst


def iso_select_walker(md_mol, start_inds, steps, current_selection=None):
    """returns indices specifying a fragment of the the passed modmol molecule.
    Indices are generated by walking a set number of steps along the molecular graph starting at the given indices"""

    if not current_selection:
        current_selection = []

    neighbors = []
    for start in start_inds:
        neighbors += [n for n in list(md_mol.graph.neighbors[start]) if n not in current_selection]

    if not steps or not neighbors:
        return current_selection
    else:
        for neighb in neighbors:
            current_selection += iso_select_walker(md_mol, [neighb], steps-1, current_selection)

    current_selection += neighbors
    return list(set(current_selection))

def sphere_select(md_mol, start_inds, radius):
    """returns indices specifying a fragment of the the passed modmol molecule.
    Indices are generated by selecting all atoms within dist of the starting indiices"""
    current_selection = []
    mol_size = len(md_mol.numbers)
    for start in start_inds:
        current_selection += [i for i in range(mol_size) if i not in start_inds and i not in current_selection and md_mol.distance_matrix[start,i] < radius]
    current_selection.sort()
    return current_selection

def cone_transf_m(apex,L,r_y):
    import numpy as np
    import transformations
    """a matrix that rotates and translates coordinates. A cone defined by the coordinates of it's apex,
    the vector L between the centre of its base and the apex, and a vector r_y between the centre of its base and a point of maximum radius (the cone can be elliptical)
    is translated such that the apex is at the origin, the cone lies along the z axis and the radius of maximum width lies along the y-axis"""
    trans = -np.array(apex).reshape(3,1)

    angle1 = transformations.angle_between_vectors(L, [0,0,1])
    if angle1:
        rot_axis1 = np.cross(L,[0,0,1])
        nrot_axis1 = rot_axis1/np.linalg.norm(rot_axis1)
        rot1 = transformations.rotation_matrix(angle1, nrot_axis1)[:3,:3]
    else:
        rot1 = np.identity(3)

    r_y = rot1.dot(r_y)
    angle2 = transformations.angle_between_vectors(r_y, [0,1,0])
    if angle2:
        rot_axis2 = np.cross([0,1,0],r_y)
        nrot_axis2 = rot_axis2/np.linalg.norm(rot_axis2)
        rot2 = transformations.rotation_matrix(angle2, nrot_axis2)[:3,:3]
    else:
        rot2 = np.identity(3)

    #combine rot matrices
    rot = np.dot(rot1,rot2)

    #make rotation matrix into an affine matrix
    final_rot = np.identity(4)
    final_rot[:3,:3] = rot

    #make translation into an affine matrix
    final_trans = np.identity(4)
    final_trans[:3,3:] = trans

    #combine translation and rotation
    final = final_rot.dot(final_trans)

    return final

#see http://ask.metafilter.com/55023/How-can-I-work-out-if-a-point-xyz-is-contained-by-a-cone
# selects all atoms that lie within the cone
def cone_select(ase_mol, start_ind=None, start=None, min_radius=None, max_radius=None, min_radius_ind=None, max_radius_ind=None, apex_ind=None, apex=None):

    if start_ind:
        centre_base = ase_mol[start_ind].position
    elif start:
        centre_base = np.array(start)
    else:
        raise RuntimeError("Need to define the centre of the base of the cone - either with an atomic index or with coordinates")


    if apex_ind:
        apex = ase_mol[apex_ind].position
    elif apex:
        apex = np.array(apex)
    else:
        raise RuntimeError("Need to define the apex of the cone - either with an atomic index or with coordinates")

    length_vec = centre_base - apex
    length = np.linalg.norm(length_vec)

    if max_radius_ind:
        max_radius_v = ase_mol[max_radius_ind].position - centre_base
    elif max_radius:
        max_radius_v = np.array(max_radius)
    else:
        raise RuntimeError("Need to define the maximum radius of the cone - either with atomic indices or with coordinates")

    if min_radius_ind:
        min_radius_v = ase_mol[min_radius_ind].position - centre_base
    elif min_radius:
        min_radius_v = np.array(min_radius)
    else:
        #assume cone is circular
        min_radius_v = np.cross(max_radius_v, length_vec/length)

    min_radius = np.linalg.norm(min_radius_v)
    max_radius = np.linalg.norm(max_radius_v)

    affine_m = cone_transf_m(apex,length_vec,max_radius_v)
    transform_coords = lambda e: affine_m.dot(e.tolist()+[1])[:3]

    in_cone = lambda coords: 0 <= coords[2] <= length and abs(coords[0]) <= coords[2]*min_radius/length and abs(coords[1]) <= coords[2]*max_radius/length

    mol_size = len(ase_mol)
    current_selection = [apex_ind]
    current_selection += [i for i in range(mol_size) if i not in current_selection and in_cone(transform_coords(ase_mol[i].position))]
    current_selection.sort()
    return current_selection


def get_active_path():
    local_home = os.path.realpath(config.get('ase', 'ase_home'))

    try:
        path = os.getcwd().split(local_home)[1]
    except IndexError:
        raise RuntimeError('Not running from within ase_home directory')
    return path


import uuid
def run_on_server(func_master, *args, **kwargs):
    """Runs the given object - a function or class with a .start() method - on the pbs server
    args/kwargs are passed to the function"""

    from gausspy.gaussian_job_manager import Job
    #we can pass in job_obj details with the function/class if we wish
    try:
        func_obj, job_obj = func_master
    except TypeError:
        func_obj = func_master
        job_obj = None

    try:
        name = func_obj.calc.label
    except AttributeError:
        try:
            name = func_obj.func_name
        except AttributeError:
            name = 'unknown'

    name += '_' + str(uuid.uuid1())

    with open(name + '.pkl', 'w') as f:
        pickle.dump([func_obj, args, kwargs], f)

    serv_home = config.get('gaussian', 'gauss_home')
    path = serv_home + get_active_path() + '/'
    #serv_work = config.get('gaussian', 'gauss_scratch')
    #path = serv_work + get_active_path() + '/'

    #gaussian uses GAUSS_SCRDIR to set the running directory, since we move to the home dir to run ase
    #we need to set this shell var to the initial location in /tmp that qsub assigns us otherwise Gaussian
    #ends up running out of home and the readwrite files mean we quickly go over our disk quota
    exec_command = 'export GAUSS_SCRDIR=`pwd`;cd {pth}; $WORK/bin/execute_func.py {f_pckl}'.format(
        pth=path,
        f_pckl=path + name + '.pkl')

    if not job_obj:
        try:
            nodes = func_obj.calc.job_params['nodes']
            mem = func_obj.calc.job_params['memory'] + nodes * 150
            time = func_obj.calc.job_params['time']
            queue = func_obj.calc.job_params['queue']

            job_obj = Job(procs=nodes, memory=mem, walltime=time, queue=queue)
        except AttributeError:
            job_obj = Job()

    script = job_obj.gen_header() + exec_command

    with open(name + '_job_script.sh', 'w') as f:
        f.write(script)

    return remote.qsub(os.getcwd() + '/' + name + '_job_script.sh', extra_files=[name + '.pkl'])
