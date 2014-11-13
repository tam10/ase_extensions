__author__ = 'clyde'

import copy
import ase.atoms
import ase.calculators.gaussian

#I was going to adopt composition to construct modified classes for ase, cclib, molmod, and other libraries I'm using.
#This would allow me to keep up-to-date with the parent libraries and keep track of my own code.
#Composition was chosen rather than inheritance because it seems to be regarded as preferable and is slightly safer
#(no name space collisions mean no accidental overwriting private methods used by public methods of the original class)

#HOWEVER ipython does not currently support tab completion for composed classes
#so I'm going to have to go with inheritance


class Atoms(ase.atoms.Atoms):
    def __init__(self, *args, **kwargs):
        try:
            self._atom_types = kwargs.pop('types')
        except KeyError:
            self._atom_types = None

        super(Atoms, self).__init__(*args, **kwargs)

    def __sub__(self, other):
        indices_to_cut = []
        frag = copy.deepcopy(self)

        for i,atom in enumerate(self):
            for other_atom in other:
                if atom == other_atom:
                    indices_to_cut.append(i)
                elif (atom.position == other_atom.position).all() and atom.symbol == other_atom.symbol:
                    frag[i].charge -= other_atom.charge
                elif (atom.position == other_atom.position).all() and atom.symbol != other_atom.symbol:
                    raise RuntimeError("Cannot subtract one element from another")


        for i in indices_to_cut:
            frag[i].cut_reference_to_atoms()
        del frag[indices_to_cut]

        return frag

    @property
    def types(self):
        return self._atom_types

    @types.setter
    def types(self, lst_types):
        self._atom_types = lst_types


#todo put all my additions to the gaussian calculator into this wrapping class
class Gaussian(ase.calculators.gaussian.Gaussian):
    def _get_mol_details(self, atoms):
        """returns the atomic configuration of the gaussian input file"""

        if 'allcheck' in self.route_self_params['geom'].lower():
            return ''

        if 'oniom' in self.route_str_params['method']:
            return self._get_oniom_details(atoms)

        mol_details = ''

        charge = sum(atoms.get_charges())
        mol_details += '%i %i\n' % (charge, self.multiplicity)

        if 'check' in self.route_self_params['geom'].lower():
            return mol_details

        symbols = atoms.get_chemical_symbols()
        coordinates = atoms.get_positions()

        for i in range(len(atoms)):
            mol_details += '%-10s' % symbols[i]
            for j in range(3):
                mol_details += '%20.10f' % coordinates[i, j]
            mol_details += '\n'
        mol_details += '\n'

        return mol_details