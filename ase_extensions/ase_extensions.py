__author__ = 'clyde'

import copy
import ase.atoms
import ase.calculators.gaussian
import numpy as np
from ase.data import atomic_numbers, chemical_symbols, atomic_masses
from ase.io import read as old_read
import warnings
from ase_utils import to_molmod


#I was going to adopt composition to construct modified classes for ase, cclib, molmod, and other libraries I'm using.
#This would allow me to keep up-to-date with the parent libraries and keep track of my own code.
#Composition was chosen rather than inheritance because it seems to be regarded as preferable and is slightly safer
#(no name space collisions mean no accidental overwriting private methods used by public methods of the original class)

#HOWEVER ipython does not currently support tab completion for composed classes
#so I'm going to have to go with inheritance


float_epsilon = 0.0000001 

class Atom(ase.atom.Atom):       
    """Class for representing a single atom.

    Parameters:
    
    symbol: str or int
        Can be a chemical symbol (str) or an atomic number (int).
    position: sequence of 3 floats
        Atom position.
    tag: int
        Special purpose tag.
    momentum: sequence of 3 floats
        Momentum for atom.
    mass: float
        Atomic mass in atomic units.
    magmom: float or 3 floats
        Magnetic moment.
    charge: float
        Atomic charge.
    """
    __slots__ = ['data', 'atoms', 'index','names']
    
    def __init__(self, symbol='X', position=(0, 0, 0),
                 tag=None, momentum=None, mass=None,
                 magmom=None, charge=None,
                 atoms=None, index=None, *args, **kwargs):     
        
        self.names = {'position': ('positions', np.zeros(3)),
                      'number':   ('numbers',   0),
                      'tag':      ('tags',      0),
                      'momentum': ('momenta',   np.zeros(3)),
                      'mass':     ('masses',    None),
                      'magmom':   ('magmoms',   0.0),
                      'charge':   ('charges',   0.0)
         }
        
        d=self.data={}        
        
        if atoms is None:
            if isinstance(symbol, str):
                d['number'] = atomic_numbers[symbol]
            else:
                d['number'] = symbol
            d['position'] = np.array(position, float)
            d['tag'] = tag
            if momentum is not None:
                momentum = np.array(momentum, float)
            d['momentum'] = momentum
            d['mass'] = mass
            if magmom is not None:
                magmom = np.array(magmom, float)
            d['magmom'] = magmom
            d['charge'] = charge

        self.index = index
        self.atoms = atoms
    def atomproperty(name, doc):
        """Helper function to easily create Atom attribute property."""
    
        def getter(self):
            return self.get(name)
    
        def setter(self, value):
            self.set(name, value)
    
        def deleter(self):
            self.delete(name)
    
        return property(getter, setter, deleter, doc)
    
    
    def xyzproperty(index):
        """Helper function to easily create Atom XYZ-property."""
    
        def getter(self):
            return self.position[index]
    
        def setter(self, value):
            self.position[index] = value
    
        return property(getter, setter, doc='XYZ'[index] + '-coordinate')
    
    def __repr__(self):
        s = "ASE Extensions Atom('%s', %s" % (self.symbol, list(self.position))
        for name in ['tag', 'momentum', 'mass', 'magmom', 'charge']:
            value = self.get_raw(name)
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                s += ', %s=%s' % (name, value)
        if self.atoms is None:
            s += ')'
        else:
            s += ', index=%d)' % self.index
        return s

    def __eq__(self, other):

        """Check for identity of two atom objects.

        Identity means: same positions, atomic numbers, and charges"""

        try:
            a = self
            b = other
            d_pos = a.position-b.position

            return ((abs(d_pos) < np.array([float_epsilon, float_epsilon, float_epsilon])).all() and
                    (a.number == b.number) and
                    (a.charge == b.charge))
        except AttributeError:
            return NotImplemented

    def cut_reference_to_atoms(self):
        """Cut reference to atoms object."""
        for name in self.names:
            self.data[name] = self.get_raw(name)
        self.index = None
        self.atoms = None
        
    def get_raw(self, name):
        """Get attribute, return None if not explicitely set."""
        if name == 'symbol':
            return chemical_symbols[self.get_raw('number')]

        if self.atoms is None:
            return self.data[name]
        
        plural = self.names[name][0]
        if plural in self.atoms.arrays:
            return self.atoms.arrays[plural][self.index]
        else:
            return None

    def get(self, name):
        """Get attribute, return default if not explicitely set."""
        value = self.get_raw(name)
        if value is None:
            if name == 'mass':
                value = atomic_masses[self.number]
            else:
                value = self.names[name][1]
        return value

    def set(self, name, value):
        """Set attribute."""
        if name == 'symbol':
            name = 'number'
            value = atomic_numbers[value]

        if self.atoms is None:
            assert name in self.names
            self.data[name] = value
        else:
            plural, default = self.names[name]
            if plural in self.atoms.arrays:
                array = self.atoms.arrays[plural]
                if name == 'magmom' and array.ndim == 2:
                    assert len(value) == 3
                array[self.index] = value
            else:
                if name == 'magmom' and np.asarray(value).ndim == 1:
                    array = np.zeros((len(self.atoms), 3))
                elif name == 'mass':
                    array = self.atoms.get_masses()
                else:
                    default = np.asarray(default)
                    array = np.zeros((len(self.atoms),) + default.shape,
                                     default.dtype)
                array[self.index] = value
                self.atoms.new_array(plural, array)

    def delete(self, name):
        """Delete attribute."""
        assert self.atoms is None
        assert name not in ['number', 'symbol', 'position']
        self.data[name] = None
    
        

    symbol = atomproperty('symbol', 'Chemical symbol')
    number = atomproperty('number', 'Atomic number')
    position = atomproperty('position', 'XYZ-coordinates')
    tag = atomproperty('tag', 'Integer tag')
    momentum = atomproperty('momentum', 'XYZ-momentum')
    mass = atomproperty('mass', 'Atomic mass')
    magmom = atomproperty('magmom', 'Initial magnetic moment')
    charge = atomproperty('charge', 'Atomic charge')
    x = xyzproperty(0)
    y = xyzproperty(1)
    z = xyzproperty(2)

    def _get(self, name):
        """Helper function for deprecated get methods."""
        warnings.warn('Use atom.%s' % name, stacklevel=3)
        return getattr(self, name)

    def _set(self, name, value):
        """Helper function for deprecated set methods."""
        warnings.warn('Use atom.%s = ...' % name, stacklevel=3)
        setattr(self, name, value)
    
    def get_symbol(self): return self._get('symbol')
    def get_atomic_number(self): return self._get('number')
    def get_position(self): return self._get('position')
    def get_tag(self): return self._get('tag')
    def get_momentum(self): return self._get('momentum')
    def get_mass(self): return self._get('mass')
    def get_initial_magnetic_moment(self): return self._get('magmom')
    def get_charge(self): return self._get('charge')
 
    def set_symbol(self, value): self._set('symbol', value)
    def set_atomic_number(self, value): self._set('number', value)
    def set_position(self, value): self._set('position', value)
    def set_tag(self, value): self._set('tag', value)
    def set_momentum(self, value): self._set('momentum', value)
    def set_mass(self, value): self._set('mass', value)
    def set_initial_magnetic_moment(self, value): self._set('magmom', value)
    def set_charge(self, value): self._set('charge', value) 
        
class Atoms(ase.atoms.Atoms):
    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None, scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None):
        
        super(Atoms, self).__init__(symbols,
                 positions, numbers,
                 tags, momenta, masses,
                 magmoms, charges,
                 scaled_positions,
                 cell, pbc, celldisp,
                 constraint,
                 calculator,
                 info)
    
    _bonds=[]
    _neighbours=[]
        
    
        
    def calculate_neighbours(self):
        """Adds an array. For each atom, a sorted list is created corresponding to the atoms it is connected to. 
        Can be called to recalculate neighbours if changes are made to the Atoms object."""
        mm_atoms = to_molmod(self)
        neighbours = mm_atoms.graph.neighbors
        neighbours_arr = [sorted(list(value)) for value in neighbours.itervalues()]
        self._neighbours = neighbours_arr
    
    def get_neighbours(self):
        if not self._neighbours:
            self.calculate_neighbours()
        return self._neighbours
        
    def calculate_bonds(self):
        """Returns a list of lists corresponding to each bond in the Atoms object."""
        neighbours=self.get_neighbours()
        bonds=[a for b in [[[a,b] for b in sorted(neighbours[a]) if a<b] for a in range(len(neighbours))] for a in b]
        self._bonds=bonds
    
    def get_bonds(self):
        if not self._bonds:
            self.calculate_bonds()
        return self._bonds
        
    def expand_selection(self,current_selection=None,mode='bonds',expansion=1,inclusive=True):
        """Returns a list corresponding to the indices of the atoms that satisfy an expansion criterion.
        current_selection:  List of atoms in self to be expanded
        using:
            'bonds':        Expand selection by {expansion (integer)} bonds
            'distances':    Expand selection by {expansion (float)} Angstroms
        inclusive:          Also returns original atom indices"""
        
        if mode=='bonds':
            
            neighbours=self.get_neighbours()
            expanded_selection=copy.deepcopy(current_selection)
            
            for i in range(expansion):
                expanded_selection=[i for i,n in enumerate(neighbours) if i in expanded_selection or any([(a in expanded_selection) for a in n])]
                
        elif mode=='distances':
            
            neighbours=self.get_neighbours()
            positions=self.get_positions()
            selected_atoms=self[current_selection]
            expanded_selection=copy.deepcopy(current_selection)
            
            for a in selected_atoms:
                for i,p in enumerate(positions):
                    if np.linalg.norm(a.position-p)<expansion:
                        expanded_selection+=[i]
        else:
            raise RuntimeError("mode must be either 'bonds' or 'distances'")
            
        if inclusive:
            return expanded_selection
        else:
            expanded_selection=[r for r in expanded_selection if not r in current_selection]
            return expanded_selection            
            
    def __sub__(self, other):
        indices_to_cut = []
        frag = copy.deepcopy(self)
        
        if isinstance(other,Atom):
            other=[other]

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
        
        if self._neighbours:
            new_neighbours=[[n for n in a if n not in indices_to_cut] for j,a in enumerate(self._neighbours) if j not in indices_to_cut]   
            frag._neighbours=new_neighbours
        
        return frag
        
    def __repr__(self):
        return 'ASE Extensions '+ase.Atoms.__repr__(self)

    def __getitem__(self, i):
        if isinstance(i, int):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')

            return Atom(atoms=self, index=i)
        else:
            ase_atoms = ase.Atoms.__getitem__(self,i)
            if self._neighbours:
                #renumber neighbours with map
                old_neighbours=self._neighbours
                new_neighbours=[sorted([i.index(n) for n in old_neighbours[a] if n in i]) for j,a in enumerate(i)]
                ase_atoms._neigbours=new_neighbours
            return(ase_atoms)
    
    def __delitem__(self, i):
        ase.Atoms.__delitem__(self,i)
        old_neighbours=self._neighbours
        if old_neighbours:
            k=[j for j in range(len(self)) if j not in i]
            new_neighbours=[sorted([k.index(n) for n in old_neighbours[a] if n not in k]) for j,a in enumerate(k)]      
            self._neighbours=new_neighbours
        
def read(fileobj, index=None, format=None):
    """Overrides the old read to generate an ASE Extensions Atoms object"""
    return Atoms(old_read(fileobj, index, format))
    
#todo put all my additions to the gaussian calculator into this wrapping class
class Gaussian(ase.calculators.gaussian.Gaussian):
    def _get_mol_details(self, atoms):
        """returns the atomic configuration of the gaussian input file"""

        if 'allcheck' in self.route_self_params['geom'].lower():
            return ''

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

        