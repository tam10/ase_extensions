__author__ = 'clyde'

def to_molmod(ase_atoms):
    import molmod, tempfile, os
    from ase.io import write

    fd, xyz_f = tempfile.mkstemp('.' + 'xyz', 'ase-')
    fd = os.fdopen(fd, 'w')
    write(fd, ase_atoms, format='xyz')
    fd.close()

    mol = molmod.Molecule.from_file(xyz_f)
    os.remove(xyz_f)

    mol.set_default_graph()
    mol.set_default_symbols()

    return mol