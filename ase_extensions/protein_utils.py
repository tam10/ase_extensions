# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:10:04 2016

@author: tam10
"""
import ase_extensions as ae
import os
from ase.atoms import default
import numpy as np
from ase.io import read as old_read
from ase.data import atomic_numbers
import copy
import random
import string
import warnings
from IPython.display import display
from IPython.html.widgets import FloatProgress

try:
    from chemview import MolecularViewer, enable_notebook
    enable_notebook()
except ImportError:
    warnings.warn(RuntimeWarning("chemview not imported. Viewer not available"))
    
class Atom(ae.Atom):   
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
    atom_type: string
        PDB Atom type.
    amber: string
        Amber atom type.
    pdb: string
        PDB atom type.
    residue: string
        PDB residue.
    resnum: long
        PDB residue index.
    chain: string
        Protein chain identifier.
    amber_charge: float
        Amber charge parameter.
    radius: float
        Atomic radius.
    """
    __slots__ = ['data', 'atoms', 'index','names']
    
    def __init__(self, symbol='X', position=(0, 0, 0),
                 tag=None, momentum=None, mass=None,
                 magmom=None, charge=None,
                 atoms=None, index=None, atom_type=None, 
                 amber=None, pdb=None, residue=None, 
                 resnum=0, chain=None, amber_charge=None,
                 radius=None, *args, **kwargs):  
                     
        super(Atom, self).__init__(symbol=symbol, position=position,
                 tag=tag, momentum=momentum, mass=mass,
                 magmom=magmom, charge=charge,
                 atoms=atoms, index=index)
        
        self.names['atom_type']     = ('atom_types',    None)
        self.names['amber']         = ('ambers',        None)
        self.names['pdb']           = ('pdbs',          None)
        self.names['residue']       = ('residues',      None)
        self.names['resnum']        = ('resnums',       0)
        self.names['chain']         = ('chains',        None)
        self.names['amber_charge']  = ('amber_charges', 0.0)
        self.names['radius']        = ('radii',         0.0)
        
        d=self.data
        
        if atoms is None:
            d['atom_type']      = atom_type
            d['amber']          = amber
            d['pdb']            = pdb
            d['residue']        = residue
            d['resnum']         = resnum
            d['chain']          = chain
            d['amber_charge']   = amber_charge
            d['radius']         = radius


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
            
    atom_type       = atomproperty('atom_type',     'PDB Atom Type')
    amber           = atomproperty('amber',         'Amber Atom Type')
    pdb             = atomproperty('pdb',           'PDB Atom Type')
    residue         = atomproperty('residue',       'PDB Residue')
    resnum          = atomproperty('resnum',        'PDB Residue Number')
    chain           = atomproperty('chain',         'PDB Chain')
    amber_charge    = atomproperty('amber_charge',  'Amber Charge')
    radius          = atomproperty('radius',        'Atomic Radius')

    def get_atom_type(self):        return self._get('atom_type')
    def get_amber(self):            return self._get('amber')    
    def get_pdb(self):              return self._get('pdb')    
    def get_residue(self):          return self._get('residue')    
    def get_resnum(self):           return self._get('resnum')
    def get_chain(self):            return self._get('chain')
    def get_amber_charge(self):     return self._get('amber_charge')
    def get_radius(self):           return self._get('radius')
    
    def set_atom_type(self, value):     self._set('atom_type',value)
    def set_amber(self, value):         self._set('amber', value)    
    def set_pdb(self, value):           self._set('pdb', value)    
    def set_residue(self, value):       self._set('residue', value)    
    def set_resnum(self, value):        self._set('resnum', value)    
    def set_chain(self, value):         self._set('chain', value)  
    def set_amber_charge(self, value):  self._set('amber_charge', value)
    def set_radius(self, value):        self._set('radius', value)
    
    def __repr__(self):
        s = "ASE Protein Atom('%s', %s" % (self.symbol, list(self.position))
        for name in ['tag', 'momentum', 'mass', 'magmom', 'charge', 'atom_type', 'amber', 'pdb', 'residue', 'resnum', 'chain','amber_charge','radius']:
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


class Atoms(ae.Atoms):
    """ASE Extensions Atoms object with functionality to handle proteins"""
    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None, atom_types=None,
                 ambers=None, pdbs=None, residues=None,
                 resnums=None, chains=None, amber_charges=None,  
                 radii=None, scaled_positions=None,
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
        
        atoms = None        
        
        if hasattr(symbols, 'GetUnitCell'):
            from ase.old import OldASEListOfAtomsWrapper
            atoms = OldASEListOfAtomsWrapper(symbols)
            symbols = None
        elif hasattr(symbols, 'get_positions'):
            atoms = symbols
            symbols = None
        if (isinstance(symbols, (list, tuple)) and
              len(symbols) > 0 and isinstance(symbols[0], Atom)):
            # Get data from a list or tuple of Atom objects:
            # Note that the order matters - must reflect __init__
            data = [[atom.get_raw(name) for atom in symbols]
                    for name in
                    ['position', 'number', 'tag', 'momentum',
                     'mass', 'magmom', 'charge', 'atom_type',
                     'amber','pdb','residue', 'resnum', 
                     'chain', 'amber_charge', 'radius']]
            atoms = self.__class__(None, *data)
            symbols = None
            
        if atoms is not None:
            if atom_types is None and atoms.has('atom_types'):
                atom_types = atoms.get_atom_types()
            if ambers is None and atoms.has('ambers'):
                ambers = atoms.get_ambers()
            if pdbs is None and atoms.has('pdbs'):
                pdbs = atoms.get_pdbs()
            if residues is None and atoms.has('residues'):
                residues = atoms.get_residues()
            if resnums is None and atoms.has('resnums'):
                resnums = atoms.get_resnums()
            if chains is None and atoms.has('chains'):
                chains = atoms.get_chains()
            if amber_charges is None and atoms.has('amber_charges'):
                amber_charges = atoms.get_amber_charges()
            if radii is None and atoms.has('radii'):
                radii = atoms.get_radii()
                
        self.set_atom_types(default(atom_types,None))
        self.set_ambers(default(ambers,None))
        self.set_pdbs(default(pdbs,None))
        self.set_residues(default(residues,None))
        self.set_resnums(default(resnums,None))
        self.set_chains(default(chains,None))
        self.set_amber_charges(default(amber_charges,None))
        self.set_radii(default(radii,None))
        self.topology={}
        self.residue_dict={}
            
        
    def set_atom_types(self, atom_types=None):
        if atom_types is None:
            self.set_array('atom_types', None)
        else:
            atom_types = np.asarray(atom_types)
            self.set_array('atom_types', atom_types, 'S50', atom_types.shape[1:])

    def get_atom_types(self):
        if 'atom_types' in self.arrays:
            return self.arrays['atom_types'].copy()
        else:
            return np.zeros(len(self),dtype='S50')
        
    def set_ambers(self, ambers=None):
        if ambers is None:
            self.set_array('ambers', None)
        else:
            ambers = np.asarray(ambers)
            self.set_array('ambers', ambers, 'S5', ambers.shape[1:])

    def get_ambers(self):
        if 'ambers' in self.arrays:
            return self.arrays['ambers'].copy()
        else:
            return np.zeros(len(self),dtype='S5')
            
    def calculate_ambers_pdbs(self,calculate_ambers=True,calculate_pdbs=True,debug=False):
        filename=self.info.get("name")
        if not filename:
            filename="".join(random.choice(string.ascii_lowercase+string.digits) for i in range(6))
        dirname=filename+"_antechamber"+os.sep
        orig=filename+'_orig.pdb'
        new=filename+'_new.mol2'
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            
        os.chdir(dirname)
        temp_atoms=self.take()
        temp_atoms.write_pdb(orig)
        os.system('antechamber -fi pdb -i {i} -fo mol2 -o {o} -at amber'.format(i=orig,o=new))
        amber_pdb_atoms=read_mol2(new,atom_col_1='pdb',atom_col_2='amber')
        
        os.chdir("..")
        if not debug:
            os.system("rm -r {d}".format(d=dirname))
        
        if calculate_ambers:
            self.set_ambers(amber_pdb_atoms.get_ambers())
        if calculate_pdbs:
            self.set_pdbs(amber_pdb_atoms.get_pdbs())
        
        
    def set_pdbs(self, pdbs=None):
        if pdbs is None:
            self.set_array('pdbs', None)
        else:
            pdbs = np.asarray(pdbs)
            self.set_array('pdbs', pdbs, 'S5', pdbs.shape[1:])

    def get_pdbs(self):
        if 'pdbs' in self.arrays:
            return self.arrays['pdbs'].copy()
        else:
            return np.zeros(len(self),dtype='S5')
        
    def set_residues(self, residues=None):
        if residues is None:
            self.set_array('residues', None)
        elif isinstance(residues,str):
            arr = np.zeros(len(self),dtype='S5')
            arr.fill(residues)
            self.set_array('residues', arr, 'S5', arr.shape[1:])
        else:
            residues = np.asarray(residues)
            self.set_array('residues', residues, 'S5', residues.shape[1:])

    def get_residues(self):
        if 'residues' in self.arrays:
            return self.arrays['residues'].copy()
        else:
            return np.zeros(len(self),dtype='S5')
        
    def set_resnums(self, resnums=None):
        if resnums is None:
            self.set_array('resnums', None)
        elif isinstance(resnums,int):
            arr = np.zeros(len(self),dtype=int)
            arr.fill(resnums)
            self.set_array('resnums', arr, int, arr.shape[1:])
        else:
            resnums = np.asarray(resnums)
            self.set_array('resnums', resnums, int, resnums.shape[1:])

    def get_resnums(self):
        if 'resnums' in self.arrays:
            return self.arrays['resnums'].copy()
        else:
            return np.zeros(len(self),dtype=int)     
        
    def set_chains(self, chains=None):
        if chains is None:
            self.set_array('chains', None)
        else:
            chains = np.asarray(chains)
            self.set_array('chains', chains, 'S5', chains.shape[1:])

    def get_chains(self):
        if 'chains' in self.arrays:
            return self.arrays['chains'].copy()
        else:
            return np.zeros(len(self),dtype='S5')  

    def set_amber_charges(self, amber_charges=None):
        if amber_charges is None:
            self.set_array('amber_charges', None)
        else:
            self.set_array('amber_charges', amber_charges, float, ())

    def get_amber_charges(self):
        if 'amber_charges' in self.arrays:
            return self.arrays['amber_charges'].copy()
        else:
            return np.zeros(len(self))  

    def set_radii(self, radii=None):
        if radii is None:
            self.set_array('radii', None)
        else:
            self.set_array('radii', radii, float, ())

    def get_radii(self):
        if 'radii' in self.arrays:
            return self.arrays['radii'].copy()
        else:
            return np.zeros(len(self))
            
    def merge_residues(self,residue_list,new_residue_name):
        """Joins a list of residues, renames and renumbers. Higher residues are renumbered."""
        for i,a in enumerate(self):
            if a.resnum in residue_list:
                a.residue=new_residue_name
                a.resnum=residue_list[0]
        self.reorder_residues()
        
    def renumber_residues(self):
        """Renumbers all residues starting from 1. Returns a dictionary with mappings."""
        renumbered={b:i+1 for i,b in enumerate([a.resnum for i,a in enumerate(self) if self[i-1].resnum!=a.resnum or i==0])}
        for atom in self:
            atom.resnum=renumbered[atom.resnum]
        self.residue_dict=None
        return renumbered
        
    def reorder_residues(self):
        resnums=self.get_resnums()
        temp=Atoms()
        [[temp.append(self[i]) for i,a in enumerate(resnums) if a==r] for r in sorted(list(set(resnums)))]
        self.__init__(temp)
    
    def calculate_topology(self,names='pdbs'):
        
        self.topology["atom_types"]=self.get_chemical_symbols()
        bonds=[tuple(b) for b in self.get_bonds()]
        if bonds:
            self.topology["bonds"]=bonds
        if names=='pdbs':
            atom_names=self.get_pdbs()
        elif names=='ambers':
            atom_names=self.get_ambers()
        else:
            raise RuntimeError("names not understood. Use 'pdbs' or 'ambers'")
        if any(atom_names):
            self.topology["atom_names"]=atom_names
        res_dict=self.get_residue_dict()
        self.topology["residue_indices"]=[tuple([n for n in range(len(self)) if self[n].resnum==r]) for r in res_dict.keys()]
        self.topology["residue_types"]=res_dict.values()
        
    def get_topology(self):
        """Returns a dictionary for ChemView with the following keys:
        KEY                 ASE PROTEIN EQUIVALENT
        atom_types:         get_chemical_symbols()
        bonds:              [tuple(b) for b in self.get_bonds()]
        atom_names:         
            names='pdbs':   get_pdbs()
            names='ambers': get_ambers()
        residue_indices:    [(n for n in range(len(self)) if self[n]==r) for r in res_dict.keys()]
        residue_types:      get_residue_dict().values()"""
        if not self.topology:
            self.calculate_topology()
        return self.topology    
        
        
    def expand_selection(self,current_selection=None,mode='bonds',expansion=1,inclusive=True):
        """mode:
            'bonds':                Expand selection by {expansion (integer)} bonds
                current_selection:  List of atom indices
                Returns:            List of atom indices
            'distances':            Expand selection by {expansion  (float) } Angstroms
                current_selection:  List of atom indices
                Returns:            List of atom indices
            'resnums':              Expand selection by {expansion (integer)} residue numbers
                current_selection:  List of residue numbers
                Returns:            List of residue numbers
        inclusive:          Also returns original atom indices/residue numbers"""
        
        if mode=='resnums':
            expanded_selection=copy.deepcopy(current_selection)
            for i in range(expansion):
                for j in copy.deepcopy(expanded_selection):
                    expanded_selection+=(j-1 not in expanded_selection)*[j-1]
                    expanded_selection+=( j  not in expanded_selection)*[ j ]
                    expanded_selection+=(j+1 not in expanded_selection)*[j+1]
            if inclusive:
                return expanded_selection
            else:
                expanded_selection=[r for r in expanded_selection if not r in current_selection]
                return expanded_selection
        else:
            return ae.Atoms.expand_selection(self,
                                            current_selection=current_selection,
                                            mode=mode,
                                            expansion=expansion,
                                            inclusive=inclusive)
    def calculate_residue_dict(self):
        resnums=list(set(self.get_resnums()))
        self.residue_dict={resnum:self.take(resnums=resnum)[0].residue for resnum in resnums}
        
    def get_residue_dict(self):
        """Returns a dictionary of mappings between residue numbers and residue names."""
        if not self.residue_dict:
            self.calculate_residue_dict()
        return self.residue_dict
        
    def get_ace_cap_sites(self, force=False):
        """Returns a list of atom indices whose PDB names are 'N' and do not have a neighbouring 'C'.
        force=True should be used if the backbone in a residue to cap is longer than 3 atoms (merged)"""
        neighbours=self.get_neighbours()
        if force:
            stripped=["".join(s for s in a.pdb if not s.isdigit()) for a in self]
            return [i for i,a in enumerate(stripped) if a=='N' if not any([b=='C' for j,b in enumerate(stripped) if j in list(neighbours[i])])]
        else:
            return [a.index for i,a in enumerate(self) if a.pdb=='N' if not any([b.pdb=='C' for b in self[list(neighbours[i])]])]
        
    def get_nme_cap_sites(self, force=False):
        """Returns a list of atom indices whose PDB names are 'C' and do not have a neighbouring 'N'.
        force=True should be used if the backbone in a residue to cap is longer than 3 atoms (merged)"""
        neighbours=self.get_neighbours()
        if force:
            stripped=["".join(s for s in a.pdb if not s.isdigit()) for a in self]
            return [i for i,a in enumerate(stripped) if a=='C' if not any([b=='N' for j,b in enumerate(stripped) if j in list(neighbours[i])])]
        else:
            return [a.index for i,a in enumerate(self) if a.pdb=='C' if not any([b.pdb=='N' for b in self[list(neighbours[i])]])]
        
    def cap_sites(self,sites=None,force=False):
        """Adds and ACE or NME cap to sites. 
        Sites must be a list of atoms whose PDB name is either 'N' (for ACE) or 'C' (for NME).
        If sites is not provided it will be determined automatically."""
            
        data_path=get_data_path()
        
        if force:
            self.set_pdbs(["".join(s for s in a.pdb if not s.isdigit()) for a in self])
        
        if not sites:
            sites=self.get_ace_cap_sites()+self.get_nme_cap_sites()
        
        for self_a0 in sites:
            
            if self[self_a0].pdb=='N':
                cap=read_pdb(data_path+'ace_cap.pdb')
                pdb_list=['N','CA','C']
                residue_offset=-1
            elif self[self_a0].pdb=='C':
                cap=read_pdb(data_path+'nme_cap.pdb')
                pdb_list=['C','CA','O']
                residue_offset=1
            
            res=self[self_a0].resnum
            chain=self[self_a0].chain
            
            self_a1=sorted([[self.get_distance(a.index,self_a0),a.index] for a in self if a.resnum==res and a.pdb==pdb_list[1]])[0][1]
            self_a2=sorted([[self.get_distance(a.index,self_a0),a.index] for a in self if a.resnum==res and a.pdb==pdb_list[2]])[0][1]
            
            cap_a0=[a.index for a in cap if a.residue=='ALA' and a.pdb==pdb_list[0]][0]
            cap_a1=[a.index for a in cap if a.residue=='ALA' and a.pdb==pdb_list[1]][0]
            cap_a2=[a.index for a in cap if a.residue=='ALA' and a.pdb==pdb_list[2]][0]
            
            self_mask_indices=[self_a0,self_a1,self_a2]
            cap_mask_indices=[cap_a0,cap_a1,cap_a2]
            
            self.fit(cap,self_mask_indices,cap_mask_indices)
            
            append_atoms=cap.remove(residues='ALA')
            for append_atom in append_atoms:
                append_atom.resnum=res+residue_offset
                append_atom.chain=chain
                self+=append_atom
        self.calculate_neighbours()
        
    def mutate(self,resnum,new_residue,remove_hydrogens=True,force=False,verbose=False,check=False):
        """Replace a residue with a standard residue. Preserves the backbone so might not work for PRO."""
        
        data_path=get_data_path()

        old_res=self.take(resnums=resnum)   
        new_prot=self.take(indices_in_tags=True)
        
        if force:
            old_res.set_pdbs(["".join(s for s in a.pdb if not s.isdigit()) for a in self])
        
        new_res=read_pdb(data_path+new_residue+'.pdb')
        if 'CB' in new_res.get_pdbs() and 'CB' in old_res.get_pdbs():
            pdb_list=['N','CA','C','CB']
        else:
            pdb_list=['N','CA','C']
        
        
        old_mask=[[a.index for a in old_res if a.pdb==pdb][0] for pdb in pdb_list]
        new_mask=[[a.index for a in new_res if a.pdb==pdb][0] for pdb in pdb_list]
        
        chain=old_res[old_mask[0]].chain
        
        old_res.fit(new_res,old_mask,new_mask)
        
        if verbose:
            for i in range(len(pdb_list)):
                print("%-2s(%s) -> %-2s(%s) Delta = %.3fA" % 
                (new_res[new_mask[i]].pdb,
                 new_residue,
                 old_res[old_mask[i]].pdb,
                 old_res[old_mask[i]].residue,
                 np.linalg.norm(old_res[old_mask[i]].position-new_res[new_mask[i]].position)))
        
        backbone_pdbs = ['N','CA','C','O','1H','2H','HA','OC','HOC','OXT','HXT']
             
        if 'CG' in new_res.get_pdbs() and 'CG' in old_res.get_pdbs():
            dihedrals=['N','CA','CB','CG']
            old_dihedral=old_res.get_dihedral([[a.index for a in old_res if a.pdb==pdb][0] for pdb in dihedrals])
            new_dihedral=new_res.get_dihedral([[a.index for a in new_res if a.pdb==pdb][0] for pdb in dihedrals])
            angle=old_dihedral-new_dihedral
            dihedral_mask=[0 if a.pdb in backbone_pdbs else 1 for a in new_res]
            new_res.rotate_dihedral([[a.index for a in old_res if a.pdb==pdb][0] for pdb in dihedrals],angle=angle,mask=dihedral_mask)
        
                
        
        if remove_hydrogens:           
            append_atoms=new_res.remove(symbols='H',pdbs=backbone_pdbs)
        else:
            append_atoms=new_res.remove(pdbs=backbone_pdbs)
        
        side_chain=new_prot.take(resnums=resnum).remove(pdbs=backbone_pdbs)
        new_prot-=side_chain
        if verbose and side_chain:
            print("\nStripped the following atoms from Residue %d:" % old_res[old_mask[0]].resnum)
            for a in side_chain:
                print(" %-4s (%d)" % (a.pdb,a.tag))
        
        if verbose and append_atoms:
            print("\nAdded the following atoms from %s:" % new_residue)
        for append_atom in append_atoms:
            append_atom.resnum=resnum
            append_atom.chain=chain
            if verbose:
                print(" %-4s" % append_atom.pdb)
            new_prot+=append_atom
        
        for a in new_prot:
            if a.resnum==resnum:
                a.residue=new_residue
        
        
        new_prot.reorder_residues()
            
        if check:
            clashes={i:[c for c in b if new_prot[c].resnum!=resnum] for i,b in enumerate(new_prot.get_neighbours()) 
                        if new_prot[i].resnum==resnum and new_prot[i].pdb not in backbone_pdbs}
            if [a for b in clashes.values() for a in b]:
                print("\nClashes found:")
                for atom_num,clash_list in clashes.iteritems():
                    for clash in clash_list:
                        print(" %s%d-%s%d Delta = %.3fA" % (new_prot[atom_num].symbol,atom_num,new_prot[clash].symbol,clash,new_prot.get_distance(atom_num,clash)))     
            else:
                print("\nNo clashes found.")
            
        return new_prot
        
    def fit(self,other,mask_indices,other_mask_indices):
        """Performs the set of transformations on 'other' that aligns its subset to a subset of 'self'."""
        self_pos=copy.deepcopy(self[mask_indices].get_positions())
        other_pos=copy.deepcopy(other[other_mask_indices].get_positions())
        
        self_ave = np.average(self_pos,0)
        other_ave = np.average(other_pos,0)
        
        self_pos -= self_ave
        other_pos -= other_ave
        
        v, l, u = np.linalg.svd(np.dot(np.transpose(self_pos), other_pos))
        r = np.dot(v, u)
        t = self_ave - np.dot(r, other_ave)
        other.set_positions(np.dot(other.get_positions(),np.transpose(r))+t)
        
    def find_subset(self,subset,target_residues=[],return_mask=False,threshold=1e-2,atoms_to_choose=4,get_neighbouring_h=True):
        """Returns a mask or a list of indices that correspond to the atoms that match the subset by coordinates and symbol."""
        
        if isinstance(target_residues,str):
            tags,target = list(map(list, zip(*[[a.index,a] for a in self if a.residue in target_residues])))
        else:
            tags,target = list(map(list, zip(*[[a.index,a] for a in self if len(target_residues)==0 or a.resnum in target_residues])))
        
        target=Atoms(target)
        target.set_tags(tags)
        
        atomic_numbers=subset.get_atomic_numbers()
        sort_choice=sorted(range(len(atomic_numbers)), key=lambda k: -atomic_numbers[k])
        subset_pairs=[[[a,b] for j,b in enumerate(sort_choice) if j<i] for i,a in enumerate(sort_choice)]
        subset_pairs=[a for b in subset_pairs for a in b]
        
        total_tp,total_sp=[],[]
        while True:
            sp=subset_pairs.pop()
            spd=subset.get_distance(sp[0],sp[1])
            tp=[[[i,j] 
                          for j,b in enumerate(target) if b.symbol==subset[sp[1]].symbol and abs(target.get_distance(i,j)-spd)<threshold] 
                          for i,a in enumerate(target) if a.symbol==subset[sp[0]].symbol]
            tp=[a for b in tp for a in b]
            if len(tp)==1:
                [total_tp.append(p) for p in tp[0] if p not in total_tp]
                [total_sp.append(p) for p in sp if p not in total_sp]
                if len(total_tp)>3:
                    target.fit(subset,total_tp,total_sp)
                    mask=[1*any([True if np.linalg.norm(s.position-t.position)<threshold else False for s in subset]) for t in target]
                    indices=[target[i].index for i in range(len(target)) if mask[i]==1]
                    if sum(mask)==len(subset):
                        break
                    else:
                        total_tp,total_sp=[],[]
        if get_neighbouring_h:
            neighbours=target.get_neighbours()
            indices+=[a for b in [[n for n in neighbours[index] if target[n].symbol=='H'] for index in indices] for a in b]
        
        indices=[a.tag for a in target if a.index in indices]
        mask=[1 if i in indices else 0 for i in range(len(self))]
        
        if return_mask:
            return mask
        else:
            return indices
        
    def take(self,symbols=None,atom_types=None,ambers=None,pdbs=None,partial_pdbs=None,residues=None,resnums=None,chains=None,tags=None,indices_in_tags=False):
        """Returns a copy whose properties match the arguments.
        With no arguments, this method returns a deep copy.
        If atom indices from the parent are required, they can be transfered to tags with indices_in_tags=True"""
        atoms=copy.deepcopy(self)
        atom_list=range(len(self))
        
        attr_dict={'symbol': symbols, 'atom_type':   atom_types,   'amber':  ambers,
                   'pdb':    pdbs,    'residue':     residues,     'resnum': resnums,
                   'chain':  chains,  'partial_pdb': partial_pdbs, 'tag':    tags}
        
        for key, value in attr_dict.iteritems():
            if value is not None:
                if not isinstance(value, list): value = [value]
                if key == 'partial_pdb':
                    atom_list = filter(lambda i: getattr(atoms[i],'pdb').translate(None,'1234567890') in set(value),atom_list)
                else:
                    atom_list = filter(lambda i: getattr(atoms[i],key) in set(value),atom_list)
        
        atoms = atoms[atom_list]
        if indices_in_tags: atoms.set_tags(atom_list)
        return self.__class__(atoms)
        
    def remove(self,symbols=None,atom_types=None,ambers=None,pdbs=None,partial_pdbs=None,residues=None,resnums=None,chains=None,tags=None,indices_in_tags=False):
        """Returns a copy whose properties do not match the arguments.
        With no arguments, this method returns a deep copy.
        If atom indices from the parent are required, they can be transfered to tags with indices_in_tags=True"""
        atoms=copy.deepcopy(self)
        atom_list=range(len(self))
        
        attr_dict={'symbol': symbols, 'atom_type':   atom_types,   'amber':  ambers,
                   'pdb':    pdbs,    'residue':     residues,     'resnum': resnums,
                   'chain':  chains,  'partial_pdb': partial_pdbs, 'tag':    tags}
        
        for key, value in attr_dict.iteritems():
            if value is not None:
                if not isinstance(value, list): value = [value]
                if key == 'partial_pdb':
                    atom_list = filter(lambda i: getattr(atoms[i],'pdb').translate(None,'1234567890') not in set(value),atom_list)
                else:
                    atom_list = filter(lambda i: getattr(atoms[i],key) not in set(value),atom_list)
        
        atoms = atoms[atom_list]
        if indices_in_tags: atoms.set_tags(atom_list)
        return self.__class__(atoms)
        
    def get_backbone(self):
        """Returns a copy of the backbone atoms (PDB = 'C', 'CA', 'N')"""
        return self.take(partial_pdbs=['C','CA','N'])
        
    def analyse(self):
        """Performs an analysis on each chain:
        Size
        Sequence with residue number mapping."""
        
        print("Basic properties:")
        print("->Formula: %s" % self.get_chemical_formula())
        print("->Size: %d" % len(self))
        
        chain_ids=list(set(self.get_chains()))
        print("%d chains found: %s" % (len(chain_ids)," ".join(chain_ids)))
        
        for chain_id in chain_ids:
            if chain_id:
                chain=self.take(chains=chain_id,indices_in_tags=True)
            else:
                chain=self.take(indices_in_tags=True)
            
            print("Chain %s:" % chain_id)
            
            backbone=chain.get_backbone()
            res_dict=chain.remove(residues='HOH').get_residue_dict()
            
            print("->Composed of %-4d protein residues and %-4d water residues" % (len(res_dict),len(chain.take(residues='HOH'))))
            print("->Sequence:")
            print("-".join(res for res in res_dict.itervalues()))
            print("-".join("{:-^3}".format(resnum) for resnum in res_dict.iterkeys()))
            
            bn=backbone.get_neighbours()
            ends=[i for i,n in enumerate(bn) if len(n)==1]
            ns_links=[i for i,n in enumerate(bn) if len(n)>2]
            
            print("->%d terminals found:" % len(ends))
            for end in ends:
                print("-->%-5d %-4s in residue number %-4d (%s)" % (backbone[end].tag,backbone[end].pdb,backbone[end].resnum,backbone[end].residue))
            if ns_links:
                print("->Unusual linkage found:")
            for ns_link in ns_links:
                print("-->%-5d %-4s in residue number %-4d (%s)" % (backbone[ns_link].tag,backbone[ns_link].pdb,backbone[ns_link].resnum,backbone[ns_link].residue))
            print("\n")
            
    def __repr__(self):
        return ae.Atoms.__repr__(self).replace('ASE Extensions ','ASE Protein ')
        

    def __getitem__(self, i):
        if isinstance(i, int):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')

            return Atom(atoms=self, index=i)
        else:
            return ae.Atoms.__getitem__(self,i)
            
    def clean(self,debug=False):
        
        self.reorder_residues()
        self.renumber_residues()
        self.calculate_ambers_pdbs(debug=debug)
        chains=self.get_chains()
        
        for i,a in enumerate(self):
            a.chain = '' if not chains[i] or isinstance(chains[i],int) or chains[i].isdigit() else chains[i]
        
            
    def write_pdb(self, filename='', atom_col_type='', print_std=False, **kwargs):
        """Write images to PDB-file.
    
        The format is assumed to follow the description given in
        http://www.wwpdb.org/documentation/format32/sect9.html."""
        
        if print_std:
            filename="STDOUT"
        elif not filename:
            filename=self.info.get("name")+".pdb"
            if not filename:
                raise RuntimeError("filename required")        
        
        filestr=''
        if self.get_pbc().any():
            from ase.lattice.spacegroup.cell import cell_to_cellpar
            cellpar = cell_to_cellpar( self.get_cell())
            # ignoring Z-value, using P1 since we have all atoms defined explicitly
            format = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
            filestr+=format % (cellpar[0], cellpar[1], cellpar[2], cellpar[3], cellpar[4], cellpar[5])
    
        format="%-6s%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f                      %2s  \n"
        # RasMol complains if the atom index exceeds 100000. There might
        # be a limit of 5 digit numbers in this field.
        MAXNUM = 100000
        
        symbols = self.get_chemical_symbols()
        atom_types = self.get_atom_types()
        pdbs = self.get_pdbs()
        residues = self.get_residues()
        chains = self.get_chains()
        resnums = self.get_resnums()
        p = self.get_positions()
        ambers = self.get_ambers()
        natoms = len(symbols)
        
        
        if atom_col_type in ['','pdb','pdbs']:
            col='pdb types'
            atom_names=pdbs
        elif atom_col_type in ['amber','ambers']:
            col='amber types'
            atom_names=ambers
        elif atom_col_type in ['symbol','symbols']:
            col='symbols'
            atom_names=symbols        
        
        col_warn_list=[i for i,a in enumerate(atom_names) if a in ['','0',0]]
        if col_warn_list:
            warnings.warn(RuntimeWarning("Empty {m} written to {f}:\n{a}".format(m=col,a=_listrepr(col_warn_list),f=filename)))
            
        filestr+='MODEL     1\n'
        for a in range(natoms):
            x, y, z = p[a]
            
            atom_type = 'ATOM' if atom_types[a] in ['','0',0,None] else atom_types[a]
            chain = '' if chains[a] in ['','0',0,None] else chains[a]
            
            atom_name = atom_names[a]
            if len(atom_name)!=4 and len(symbols[a])==1:
                atom_name=' '+atom_name
            #                       %-6s       %5d           %-4s       %3s          %1s    %4d         %8.3f%8.3f%8.3f%2s
            filestr+=format % (atom_type, a+1 % MAXNUM, atom_name, residues[a], chain, resnums[a], x,   y,   z,   symbols[a].rjust(2))
        filestr+='ENDMDL\n'
        
        if print_std:
            print(filestr)
        else:
            with open(filename, 'w')as fileobj:
                fileobj.write(filestr)
        
    def write_mol2(self, filename='', atom_col_1='',atom_col_2='', mol_name='', mol_type='SMALL', print_std=False, **kwargs):
        """Writes images to a Sybyl mol2 file."""
        
        if print_std:
            filename="STDOUT"
        elif not filename:
            filename=self.info.get("name")+".mol2"
            if not filename:
                raise RuntimeError("filename required")
    
        if mol_name=='': mol_name=self.get_chemical_formula()
        
        bonds = self.get_bonds()        
        num_bonds = len(bonds)
        pdbs=self.get_pdbs()
        ambers=self.get_ambers()
        symbols=self.get_chemical_symbols()
        
        
        if atom_col_1 in ['','pdb','pdbs']:
            col_1='pdb types'
            atom_names=pdbs
        elif atom_col_1 in ['amber','ambers']:
            col_1='amber types'
            atom_names=ambers
        elif atom_col_1 in ['symbol','symbols']:
            col_1='symbols'
            atom_names=symbols
            
        if atom_col_2 in ['','amber','ambers']:
            col_2='amber types'
            sybyl_names=ambers
        elif atom_col_2 in ['pdb','pdbs']:
            col_2='pdb types'
            sybyl_names=pdbs
        elif atom_col_2 in ['symbol','symbols']:
            col_2='symbols'
            sybyl_names=symbols
        
        col_1_warn_list=[i for i,a in enumerate(atom_names) if a in ['','0',0]]
        if col_1_warn_list:
            warnings.warn(RuntimeWarning("Empty {m} written to {f}:\n{a}".format(m=col_1,a=_listrepr(col_1_warn_list),f=filename)))
            
        col_2_warn_list=[i for i,a in enumerate(sybyl_names) if a in ['','0',0]]
        if col_2_warn_list:
            warnings.warn(RuntimeWarning("Empty {m} written to {f}:\n{a}".format(m=col_2,a=_listrepr(col_2_warn_list),f=filename)))
                     
        
        filestr="@<TRIPOS>MOLECULE\n"
        filestr+="%s\n" % mol_name
        filestr+=" %4d %5d %5d %5d %5d\n" % (len(self),num_bonds,1,0,0)
        filestr+="%s\n" % mol_type
        filestr+="No Charge or Current Charge\n\n\n"
        filestr+="@<TRIPOS>ATOM\n"
        for i,atom in enumerate(self):
            atom_name = atom_names[i]
            sybyl_name = sybyl_names[i]
            filestr+=" %6d %-8s %9.4f %9.4f %9.4f %-6s %4d %-6s %10.6f\n" % (i+1,atom_name,atom.x,atom.y,atom.z,sybyl_name,atom.resnum,atom.residue,atom.amber_charge)
        
        
        filestr+="@<TRIPOS>BOND\n"
        for i,bond in enumerate(bonds):
            filestr+=" %5d %4d %4d %1d\n" % (i+1,bond[0]+1,bond[1]+1,1)
        
        if print_std:
            print(filestr)
        else:
            with open(filename, 'w')as fileobj:
                fileobj.write(filestr)
            
class Viewer(MolecularViewer):
    
    def __init__(self,atoms,width=500,height=500):
        
        ps=atoms.get_positions()/10
        ts=atoms.get_topology()
        MolecularViewer.__init__(self,coordinates=ps,topology=ts,width=width,height=height)
        self.server=None
        self.client=None
        self.atoms=atoms
        self._server_run=None
        self._motion=None
        self._zoom=None
        self._rotatexy=None
        self._touch_count=None
        self._touch=None
        self._vectors=None
        self._measurements=None
        self._label_gesture_pos=None
        self._axes_gesture_pos=None
        self._label_mode=0
        
    def add_overlay(self, atoms, colour_list=None, ball_and_sticks=True, ball_radius=0.05, stick_radius=0.02, opacity=1.0):
        
        ps=atoms.get_positions()/10
        ts=atoms.get_topology()
        mv=MolecularViewer(ps,topology=ts)
        
        if ball_and_sticks:
            mv.ball_and_sticks(ball_radius=ball_radius,stick_radius=stick_radius,colorlist=colour_list,opacity=opacity)
        else:
            mv.lines()
            
        for key,value in mv.representations.iteritems():
            self.add_representation(rep_type=value['rep_type'],options=value['options'],rep_id=key)
            
    def measure(self, atom_list=None, coords=None, text_size=24, text_colour=0x4444ff, dot_size=18, dot_colour=0xff3333, dot_spacing=0.03):
        
        if coords is None:
            if not isinstance(atom_list,list):
                raise RuntimeError("atom_list must be a list")
            coords=self.coordinates[atom_list]
        
        ts,ps,ss,cs,vs=[],[],[],[],[]
        
        if len(coords)==2:
            ls=[coords]
            t='%0.2f' % (np.linalg.norm(coords[0]-coords[1])*10)
            p=np.average([coords[0],coords[1]],0)
        elif len(coords)==3:
            ls=[[coords[0],coords[1]],[coords[2],coords[1]]]
            p=np.average([c for l in ls for c in l],0)
        elif len(coords)==4:
            ls=[[coords[0],coords[1]],[coords[1],coords[2]],[coords[2],coords[3]]]
            p=np.average([c for l in ls for c in l],0)
        else:
            raise RuntimeError("atom_list wrong size ({s}). Must be 2,3 or 4".format(s=len(coords)))
            
        for l in ls:
            v=l[1]-l[0]
            vl=np.linalg.norm(v)
            vn=v/vl
            vs.append(vn)
            
            for n in np.arange(0,vl,dot_spacing):
                ts.append('•')
                ps.append(l[0]+n*vn)
                ss.append(dot_size)
                cs.append(dot_colour)
                
        if len(coords)==3:
            a=np.dot(vs[0],vs[1])
            t='%0.1f°' % np.math.degrees(np.math.acos(a))
        elif len(coords)==4:
            c0=np.cross(vs[1],vs[0])
            c1=np.cross(vs[2],vs[1])
            c0/=np.linalg.norm(c0)
            c1/=np.linalg.norm(c1)
            a=np.vdot(c0,c1)
            t='%0.1f°' % np.math.degrees(np.math.acos(a))
        
        ts.append(t)
        ps.append(p)
        ss.append(text_size)
        cs.append(text_colour)
        ps=np.array(ps)
        
        self.labels(text=ts,coordinates=ps,sizes=ss,colorlist=cs)
        
    def l(self, text=None, coordinates=None, colorlist=None, sizes=None, fonts=None, opacity=1.0, mode=None, atom_list=None):
    
        def to_list(item,length):
            if not item is None:
                if isinstance(item,(int,str,float)):
                    item = [item]
                elif isinstance(item,(np.generic,np.ndarray)):
                    item = list(item)
                if len(item) == 1:
                    item *= length
            return(item)
        
        if mode is None:
            self.labels(text, coordinates, colorlist, sizes, fonts, opacity)
        else:
            if atom_list is None:
                atom_list=range(len(self.coordinates))
                
            coordinates=self.coordinates[atom_list]
            
            sizes = to_list(sizes, len(coordinates))
            text = to_list(text, len(coordinates))
            fonts = to_list(fonts, len(coordinates))
            colorlist = to_list(colorlist, len(coordinates))
            
            if mode in ['g','ng','numg','indexg','ig','indicesg']:
                text=[str(i+1) for i in atom_list]
            if mode in ['n','num','index','i','indices']:
                text=[str(i) for i in atom_list]
            elif mode in ['p','pdb','pdbs']:
                text=list(self.atoms.get_pdbs())
            elif mode in ['a','amber','ambers']:
                text=list(self.atoms.get_ambers())
            elif mode in ['t','tag','tags']:
                text=list(self.atoms.get_tags())
            elif mode in ['c','charge','charges']:
                text=list(self.atoms.get_amber_charges())
            else:
                if text is None:
                    if len(self.topology.get('atom_types'))>=len(atom_list):
                        text=[self.topology['atom_types'][i]+str(i+1) for i in atom_list]
                    else:
                        text=[str(i+1) for i in atom_list]
                
            self.labels(text, coordinates, colorlist, sizes, fonts, opacity)
            
    def b(self, ball_radius=0.05, stick_radius=0.02, colorlist=None, opacity=1.0, charges=None):
        
        if not charges is None:
            colorlist=[]
            if isinstance(charges,bool):
                charges=self.atoms.get_amber_charges()
            norm=max(abs(charges))
            for c in charges:
                colorlist.append(256*256*int(255*(c/norm)) if c>0 else int(255*(-c/norm)))
        
        
        self.ball_and_sticks(ball_radius, stick_radius, colorlist, opacity)
        
    def average(self,atom_list):
        if atom_list is None:
            coords=self.coordinates
        else:
            coords=self.coordinates[atom_list]
        return np.average(coords,0)
            
    def remote(self,remote_ip="192.168.2.4",outgoing_port=8000,incoming_port=9000,debug=False):
        import OSC, types
    
        try:
            if self.server:
                self.server.close()
        except NameError:
            pass
        self.server=OSC.OSCServer(("0.0.0.0",8000))
        try:
            if self.client:
                self.client.close()
        except NameError:
            pass
        self.client=OSC.OSCClient()
    
        def stop_button(path,tags,args,source):
            if path=="/1/stop":
                if args[0]==1:
                    self._server_run=False
        def touch_count(path,tags,args,source):
            c=path.replace("/1/multixy1/","").replace("/z","")
            if c.isdigit():
                self._touch_count[int(c)-1]=args[0]
                self._vectors[int(c)-1]=np.array([0,0])
                self._touch[int(c)-1]=np.array([0,0])
                if int(c)==3 and args[0]==1:
                    self._label_gesture_pos=np.average([t for t in self._touch],0)
                if int(c)==4 and args[0]==1:
                    self._axes_gesture_pos=np.average([t for t in self._touch],0)
        def multi(path,tags,args,source):
            c=path.replace("/1/multixy1/","")
            if c.isdigit():
                self._vectors[int(c)-1]=self._touch[int(c)-1]-np.array(args) if any(self._touch[int(c)-1]!=[0,0]) else np.array([0,0])
                self._touch[int(c)-1]=np.array(args)
        def handle_error(self,request,client_address):
            pass
        #def get_delta(self,dimension):
        #    new,old=[sum([t[dimension] for t in l]) for l in  self._touch]
        #    return (old-new)/len(self._touch[0])
        def get_vectors(self):
            return [v[1]-v[0] for v in self._touch]
        def cycle_labels(self,direction):
            #self._label_modes=["none","name_number","number","pdbs","ambers","tags"]
        
        
            msg=OSC.OSCMessage("/1/led1/color")
            map(msg.append, ["red"])
            self.client.sendto(msg, (remote_ip,incoming_port), timeout=0)
            
            self._label_mode+=direction
            if self._label_mode==-1:
                self._label_mode=5
            if self._label_mode==6:
                self._label_mode=0
            
            self.remove_labels()
            if self._label_mode==1:
                self.labels()
            elif self._label_mode==2:
                self.labels(text=[str(i+1) for i in range(len(self.coordinates))])
            elif self._label_mode==3:
                self.labels(text=list(self.atoms.get_pdbs()))
            elif self._label_mode==4:
                self.labels(text=list(self.atoms.get_ambers()))
            elif self._label_mode==5:
                self.labels(text=list(self.atoms.get_tags()))
            
            msg=OSC.OSCMessage("/1/led1/color")
            map(msg.append, ["green"])
            self.client.sendto(msg, (remote_ip,incoming_port), timeout=0)
            
        def update_debug(self):
            for i,t in enumerate(self._vectors):
                
                msg=OSC.OSCMessage("/1/l"+str(i))
                map(msg.append, [self._touch_count[i]])
                self.client.sendto(msg, (remote_ip,incoming_port), timeout=0)
                
                msg=OSC.OSCMessage("/1/x"+str(i))
                map(msg.append, ["%6.5f" % t[0]])
                self.client.sendto(msg, (remote_ip,incoming_port), timeout=0)
                
                msg=OSC.OSCMessage("/1/y"+str(i))
                map(msg.append, ["%6.5f" % t[1]])
                self.client.sendto(msg, (remote_ip,incoming_port), timeout=0)
            
    
        self.server.addMsgHandler("/1/stop",stop_button)
        self.server.addMsgHandler("/1/multixy1/1/z",touch_count)
        self.server.addMsgHandler("/1/multixy1/2/z",touch_count)
        self.server.addMsgHandler("/1/multixy1/3/z",touch_count)
        self.server.addMsgHandler("/1/multixy1/1",multi)
        self.server.addMsgHandler("/1/multixy1/2",multi)
        self.server.addMsgHandler("/1/multixy1/3",multi)
    
        self.server.handle_error=types.MethodType(handle_error,self.server)
    
        self.server.socket.settimeout(1)
        
        self._touch_count=[0]*4
        self._touch=[np.array([0]*2)]*4
        self._vectors=[np.array([0]*2)]*4
        self._server_run=True
        msg=OSC.OSCMessage("/1/led1")
        map(msg.append, [1])
        self.client.sendto(msg, (remote_ip,incoming_port), timeout=0) 
            
        msg=OSC.OSCMessage("/1/led1/color")
        map(msg.append, ["green"])
        self.client.sendto(msg, (remote_ip,incoming_port), timeout=0) 
    
        while self._server_run:
    
            self.server.handle_request()
            vs=self._vectors
            if sum(self._touch_count)==1: #Rotate
                xrot=vs[0][0]
                yrot=vs[0][1]
                self._remote_call('rotateLeft',angle=xrot)
                self._remote_call('rotateUp',angle=yrot)
            elif sum(self._touch_count)==2: #Zoom and pan
                ns=[v/np.linalg.norm(v) if np.linalg.norm(v)!=0 else np.array([0,0]) for v in vs]
                
                if np.dot(ns[0],ns[1])<-0.5: #Zoom
                    old_sep=np.linalg.norm(self._touch[1]-self._touch[0])
                    new_sep=np.linalg.norm(self._touch[1]+vs[1]-self._touch[0]-vs[0])
                    zoom=(new_sep-old_sep)/2
                    if zoom > 0:
                        self._remote_call('dollyOut',dollyScale=1+(zoom))
                    elif zoom < 0:
                        self._remote_call('dollyIn',dollyScale=1-(zoom))
                        
                elif np.dot(ns[0],ns[1])>0.5: #Pan
                    panx=np.average([vs[0],vs[1]],0)[0]*100
                    pany=np.average([vs[0],vs[1]],0)[1]*100
                    self._remote_call('pan',deltaX=panx,deltaY=pany)
            elif sum(self._touch_count)==3: #Change labels and axes
                if self._label_gesture_pos is not None:
                    current_ave_pos=np.average([t for t in self._touch],0)                   
                    if np.linalg.norm(current_ave_pos[1]-self._label_gesture_pos[1])>0.3:
                        cycle_labels(self,1)                 
                    elif np.linalg.norm(current_ave_pos[1]-self._label_gesture_pos[1])<-0.3:
                        cycle_labels(self,-1)
                if self._axes_gesture_pos is not None:
                    current_ave_pos=np.average([t for t in self._touch],0)                   
                    if abs(np.linalg.norm(current_ave_pos[1]-self._axes_gesture_pos[1]))>0.3:
                        self.toggle_axes()
            
            if debug:
                update_debug(self)
    
        msg=OSC.OSCMessage("/1/led1")
        map(msg.append, [0])
        self.client.sendto(msg, (remote_ip,incoming_port), timeout=0)
    
        self.server.close()
            
    
def get_data_path():
    return os.path.dirname(os.path.abspath(__file__))+os.sep+"Data"+os.sep
    
def read(fileobj, index=None, format=None):
    return Atoms(old_read(fileobj, index, format))
    
def read_pdb(fileobj, index=-1, alt_structure='A'):
    """Read PDB files.

    The format is assumed to follow the description given in
    http://www.wwpdb.org/documentation/format32/sect9.html."""
            
    if isinstance(fileobj, str):
        with open(fileobj,'r') as fileobj:
            lines = fileobj.readlines()
    else:
        lines = fileobj.readlines()

    data_path=get_data_path()
    with open(data_path+"ambers.txt",'r') as amberobj:
        amber_ref = [amber.split() for amber in amberobj.readlines() if '#' not in amber]        
    
    images = []
    atoms = Atoms()
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                # Atom name is arbitrary and does not necessarily contain the element symbol.
                # The specification requires the element symbol to be in columns 77+78.
                if line[16]!=' ' and line[16]!=alt_structure:
                    continue
                charge=0
                atom_type=line[0:6].strip()
                try:
                    symbol,charge = [(amber[1],amber[2]) for amber in amber_ref if amber[0]==line[76:78].strip()]
                except ValueError:
                    symbol = line[12:16].strip()[0]
                pdb = line[12:16].strip()
                words = line[30:55].split()
                position = np.array([float(words[0]), 
                                     float(words[1]),
                                     float(words[2])])
                residue = line[17:20].strip()
                chain = line[21].strip()
                resnum = int(line[22:26].strip())
                if line[78:80].strip()!='':
                    if line[79]=='+':
                        charge=int(line[78])
                    if line[79]=='-':
                        charge=-int(line[78])
                atoms.append(Atom(symbol=symbol,
                                  position=position,
                                  residue=residue,
                                  resnum=resnum,
                                  amber_charge=charge,
                                  chain=chain,
                                  pdb=pdb,
                                  atom_type=atom_type))
            except:
                pass
        if line.startswith('ENDMDL'):
            atoms.info["name"]=fileobj.name.replace(".pdb","").split(os.sep)[-1]
            images.append(atoms)
            atoms = Atoms()
    if len(images) == 0:
        atoms.info["name"]=fileobj.name.replace(".pdb","").split(os.sep)[-1]
        images.append(atoms)
    return images[index]

def read_pqr(fileobj,atom_col_type=''):    
    if isinstance(fileobj, str):
        with open(fileobj,'r') as fileobj:
            ls = [l.split() for l in fileobj.readlines() if l.split()[0]=='ATOM' or l.split()[0]=='HETATM']  
    else:
        ls = [l.split() for l in fileobj.readlines() if l.split()[0]=='ATOM' or l.split()[0]=='HETATM']  
    
    o=0 if ls[0][4].isdigit() else 1
    
    atom_types = [l[0] for l in ls]
    atom_names = [l[2] for l in ls]
    residues = [l[3] for l in ls]
    chains = [(o==1)*l[4] for l in ls]
    resnums = [int(l[4]) for l in ls]
    positions = [[float(p) for p in l[5:8]] for l in ls]
    charges = [float(l[8]) for l in ls]
    radii = [float(l[9]) for l in ls]
    
    if not atom_col_type:
        atom_col_type=amber_or_pdb(atom_names)
    
    if atom_col_type in ['pdb','pdbs']:
        pdbs=atom_names
        symbols,alt_charges=get_symbols_charges(atom_names,'pdb')
        ambers=None
    elif atom_col_type in ['amber','ambers']:
        ambers=atom_names
        symbols,alt_charges=get_symbols_charges(atom_names,'amber')
        pdbs=None
    elif atom_col_type in ['symbol','symbols']:
        symbols=atom_names
        ambers=None
        pdbs=None
        alt_charges=None
        
    if sum([abs(c) for c in charges])<1e-5:
        charges=alt_charges
    
    atoms = Atoms(symbols=symbols,positions=positions,atom_types=atom_types,pdbs=pdbs,ambers=ambers,chains=chains,amber_charges=charges,resnums=resnums,residues=residues,radii=radii)
    atoms.info["name"]=fileobj.name.replace(".pqr","").split(os.sep)[-1]
    
    return atoms
    
def read_mol2(fileobj,atom_col_1='',atom_col_2=''):
    """Read mol2 files.
    Use atom_col_1/atom_col_2 = 'pdb'/'amber' if known, otherwise they will be determined automatically.

    The format is assumed to follow the description given in
    http://www.tripos.com/data/support/mol2.pdf"""
    if isinstance(fileobj, str):
        with open(fileobj,'r') as fileobj:
            data = fileobj.read()
    else:
        data = fileobj.read()
        
    section_data = data.split('@')[1:]
    keys = [e.replace('@<TRIPOS>', '') for e in data.split() if '@' in e]
    master_dict = {k: section_data[i].split('\n')[1:-1] for i, k in enumerate(keys)}
    
    ls = [l.split() for l in master_dict.get('ATOM')]
    sybyl_atom_names = [l[1] for l in ls]
    positions = [[float(p) for p in l[2:5]] for l in ls]
    sybyl_atom_types = [l[5] for l in ls]
    resnums = [int(l[6]) for l in ls]
    residues = [l[7] for l in ls]
    charges = [float(l[8]) for l in ls]
    
    if not positions:
        raise RuntimeError("Failed to read {f}. No position data found".format(f=fileobj.name))
    
    if not atom_col_1:
        atom_col_1=amber_or_pdb(sybyl_atom_names)
    if not atom_col_2:
        atom_col_2=amber_or_pdb(sybyl_atom_types)
    
    if atom_col_1 in ['pdb','pdbs']:
        pdbs=sybyl_atom_names
        symbols,alt_charges=get_symbols_charges(pdbs,'pdb')
        if atom_col_2 in ['amber','ambers']: 
            ambers=sybyl_atom_types
            symbols=[s if s[0]!="D" else ambers[i][0] for i,s in enumerate(symbols)]  
        else:
            ambers=None
    else:
        ambers=sybyl_atom_names
        symbols,alt_charges=get_symbols_charges(sybyl_atom_names,'amber')
        if atom_col_2 in ['pdb','pdbs']:
            pdbs=sybyl_atom_types
            symbols=[s if s[0]!="D" else pdbs[i][0] for i,s in enumerate(symbols)]  
        else:
            pdbs=None
        
    if sum([abs(c) for c in charges])<1e-5:
        charges=alt_charges
      
    
    atoms=Atoms(symbols=symbols,positions=positions,pdbs=pdbs,ambers=ambers,amber_charges=charges,resnums=resnums,residues=residues)
    atoms.info["name"]=fileobj.name.replace(".mol2","").split(os.sep)[-1]
    
    return atoms
    
def read_spf(filename, structure=['A'], return_atom_names=False):
    def get_symbol(symbol_section):
        symbol = ''
        for s in symbol_section:
            if s.isalpha():
                symbol += s
            else:
                return symbol
    with open(filename,'r') as spf_obj:
        spf_lines = spf_obj.readlines()
    atoms = Atoms()
    atom_names = []
    transform = False
    for l in spf_lines:
        if l.startswith('CELL'):
            transform = True
            scl = l.split()[2:]
            abcs = [float(s) for s in scl[0:3]]
            angs = [np.deg2rad(float(s)) for s in scl[3:6]]
            v = (1 - (np.cos(angs[0]))**2 - (np.cos(angs[1]))**2 - (np.cos(angs[2]))**2 + 2*np.cos(angs[0])*np.cos(angs[1])*np.cos(angs[2]))**0.5
            trans_matrix = np.array([[abcs[0], abcs[1]*np.cos(angs[2]), abcs[2]*np.cos(angs[1])],
                                     [0, abcs[1]*np.sin(angs[2]), abcs[2]*(np.cos(angs[0])-np.cos(angs[1])*np.cos(angs[2]))/np.sin(angs[2])],
                                     [0, 0, abcs[2]*v/np.sin(angs[2])]])
            
        if l.startswith('ATOM') and ((l[8] in structure) or (l[8] == ' ')):
            symbol = get_symbol(l[5:11])
            if transform:
                position = np.dot(trans_matrix,[float(l[12:21]),float(l[21:30]),float(l[30:39])])
            else:
                position = [float(l[12:21]),float(l[21:30]),float(l[30:39])]
                
            atoms.append(Atom(symbol=symbol,position=position))
            if return_atom_names:
                atom_names.append(pdb=l[5:11].strip())
    if return_atom_names:
        return(atoms,atom_names)
    else:
        return(atoms)
    
def read_log(filename,index):
    atomic_symbols={v:k for k,v in atomic_numbers.iteritems()}
    with open(filename,'r') as lfile:
        lines=lfile.readlines()
    mols_list = []
    for n, line in enumerate(lines):
        if ('Input orientation:' in line):
            i = 0
            mol_list = []
            while ('----' not in lines[n + i + 5] ):
                info = lines[n + i + 5].split()
                symbol = atomic_symbols[int(info[1])]
                position = [float(info[3]), float(info[4]), float(info[5])]
                mol_list.append([symbol, position])
                i += 1
            mols_list.append(mol_list)
    atoms=Atoms()
    for mol in mols_list[index]:
        atoms.append(Atom(symbol=mol[0],position=mol[1]))
    return atoms
    
def amber_or_pdb(atom_name_list):
    data_path=get_data_path()
    with open(data_path+"ambers.txt","r") as ambfile:
        ambers = [amber.split()[0] for amber in ambfile.readlines() if '#' not in amber]
    with open(data_path+"pdbs.txt","r") as pdbfile:
        pdbs = [pdb.split()[0] for pdb in pdbfile.readlines() if '#' not in pdb]
        
    pdb_type=sum([(a in pdbs and not a in ambers) for a in atom_name_list])    
    amber_type=sum([(a in ambers and not a in pdbs) for a in atom_name_list])
    
    if pdb_type>amber_type:
        return('pdb')
    else:
        return('amber')
    
def _listrepr(atom_list):
    r,p=[],[]
    for i in atom_list:
        if i-1 not in atom_list:
            p=[i]
            if i+1 not in atom_list:
                r+=['{p}'.format(p=p[0])]
        elif i+1 not in atom_list:
            p+=[i]
            r+=['{s}-{e}'.format(s=p[0],e=p[1])]
    return ', '.join(r)
        
def get_symbols_charges(atom_names, atom_type=''):
    data_path=get_data_path()
    with open(data_path+"ambers.txt","r") as ambfile:
        amber_ref = [amber.split() for amber in ambfile.readlines() if '#' not in amber]
    with open(data_path+"pdbs.txt","r") as pdbfile:
        pdb_ref = [pdb.split() for pdb in pdbfile.readlines() if '#' not in pdb]        
        
    if isinstance(atom_names,str): atom_names=[atom_names]
    
    if not atom_type:
        atom_type = amber_or_pdb(atom_names)
    if atom_type=='pdb':
        ref={r[0]: r[1:] for r in pdb_ref}
    else:
        ref={r[0]: r[1:] for r in amber_ref}
    
    symbols,charges=list(map(list,zip(*[(ref.get(a) if ref.get(a) else [[c for c in a if not c.isdigit()][0],'0']) for a in atom_names])))
    
    if len(atom_names)==1:
        return symbols[0], charges[0]
    else:
        return symbols, charges
        
def peptide(sequence, phi_angles=None, psi_angles=None, omega_angles=None, degrees=True, progress=False):
    '''Generates a peptide from its three letter sequence.
    If angles are not provided, the following defaults will be used:
    phi: -57º
    psi: -47º
    omega: 0º'''
    data_path = get_data_path()
    
    if progress:
        overall = FloatProgress(min=0, max=len(sequence), description='Current ({a} {r}/{t}) : '.format(a='   ',r=0,t=len(sequence)))
        display(overall)
    
    def rest_of_residue(peptide, inc, exc):

        selection = copy.deepcopy(inc)
        resnum = peptide[inc[1]].resnum
        while True:
            new_selection = [i for i in peptide.expand_selection(selection,inclusive=False) if peptide[i].resnum == resnum and i not in exc]
            if len(new_selection)>0:
                selection += new_selection
            else:
                return selection

    def set_torsion(peptide, pdbs, res_offsets, angle, resnum):
        torsion_i = [index for j in range(4) for index, a in enumerate(peptide) if a.pdb == pdbs[j] and a.resnum == res_offsets[j]+resnum]
        inc = [torsion_i[3],torsion_i[2]]
        exc = [torsion_i[1],torsion_i[0]]
        mask = (peptide.get_resnums() > resnum ) | (np.in1d(range(len(peptide)),rest_of_residue(peptide, inc, exc)))
        peptide.set_dihedral(torsion_i, angle, mask)
        
    def get_angles(angles, name, length, default):
        if angles is None:
            angles = [default]*(len(sequence)-1)
        elif degrees:
            angles = [np.deg2rad(a) for a in angles]
        if len(angles) != length:
            raise RuntimeError('Incorrect length for {n} angles ({al}) - should be {cl}.'.format(n=name, al=len(angles), cl=length))
        return(angles)
    
    #Information required to describe the atoms involved in the three dihedral rotations in each residue
    phi_i   = [['N' , 'CA', 'C' , 'O' ], [1, 1, 1, 1]]
    psi_i   = [['C' , 'N' , 'CA', 'C' ], [0, 1, 1, 1]]
    omega_i = [['O' , 'C' , 'N' , 'CA'], [0, 0, 1, 1]]
    
    #Psis, phis and omegas are checked and converted to radians if necessary
    #If they don't exist, the defaults -57º, -47º and 0º are used respectively
    phi_angles   = get_angles(phi_angles  , 'Phi'  , len(sequence)-1, -0.995)
    psi_angles   = get_angles(psi_angles  , 'Psi'  , len(sequence)-1, -0.820)
    omega_angles = get_angles(omega_angles, 'Omega', len(sequence)-1,  0    )
    
    for resnum, s in enumerate(sequence):
        if progress:overall.description = 'Current ({a} {r}/{t}) : '.format(a=s,r=resnum+1,t=len(sequence))
            
        #Each amino acid in the sequence is loaded as an Atoms object
        aa = read_pdb(data_path+s+'.pdb')
        aa.set_resnums(resnum+1)
        
        
        #Proline contains an additional hydrogen on the nitrogen group
        if s == 'PRO':
            aa = aa.remove(pdbs='H')
        
        if resnum == 0:
            peptide = aa.take()
            
        else:
            #This is where the new N sits
            oc_pos = peptide.take(pdbs = 'OC').positions[-1]
            offset =  oc_pos - aa.take(pdbs = 'N').positions[0]
            aa.positions += offset
            
            #Terminal O-H of previous residue is removed and replaced by new amino acid
            peptide = peptide.remove(pdbs = ['OC','HOC'])
            peptide += aa
            
            #The new amino acid is rotated to be sp2 with respect to the peptide
            #Atoms on either side are selected to create a mask for rotation
            d_is = [index for j in range(4) for index, a in enumerate(peptide) if a.pdb == omega_i[0][j] and a.resnum == omega_i[1][j]+resnum]
            inc = [d_is[3],d_is[2]]
            exc = [d_is[1],d_is[0]]
            mask = (peptide.get_resnums() > resnum) | (np.in1d(range(len(peptide)),rest_of_residue(peptide, inc, exc)))
            
            angle = 2.1
            peptide.set_angle([d_is[0], d_is[1], d_is[2]], angle, mask)
            peptide.set_angle([d_is[1], d_is[2], d_is[3]], angle, mask)
            
            #Phi, psi and omega are set here
            set_torsion(peptide, phi_i[0]  , phi_i[1]  , phi_angles[resnum-1]  , resnum)
            set_torsion(peptide, psi_i[0]  , psi_i[1]  , psi_angles[resnum-1]  , resnum)
            set_torsion(peptide, omega_i[0], omega_i[1], omega_angles[resnum-1], resnum)
        if progress:
            overall.value = resnum+1
            
    return peptide
    
def _sequence_dict():
    return {'ALA': 'A', # Alanine
            'ARG': 'R', # Arginine
            'ASN': 'N', # Asparagine
            'ASP': 'D', # Aspartic Acid
            'CYS': 'C', # Cysteine
            'GLU': 'E', # Glutamic Acid
            'GLN': 'Q', # Glutamine
            'GLY': 'G', # Glycine
            'HIS': 'H', # Histidine
            'ILE': 'I', # Isoleucine
            'LEU': 'L', # Leucine
            'LYS': 'K', # Lysine
            'MET': 'M', # Methionine
            'PHE': 'F', # Phenylanaline
            'PRO': 'P', # Proline
            'SER': 'S', # Serine
            'THR': 'T', # Threonine
            'TRP': 'W', # Tryptophan
            'TYR': 'Y', # Tyrosine
            'VAL': 'V'  # Valine
            }

def convert_1_to_3(sequence):
    '''Returns the 3-letter code sequence for a 1-letter code sequence'''
    sequence_dict = {v:k for k,v in _sequence_dict().iteritems()}
    if isinstance(sequence,str):
        sequence = list(sequence)
    return [sequence_dict.get(s) for s in sequence]
    
def convert_3_to_1(sequence):
    '''Returns the 1-letter code sequence for a 3-letter code sequence'''
    sequence_dict = _sequence_dict()
    return [sequence_dict.get(s) for s in sequence]