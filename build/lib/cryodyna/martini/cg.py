import os
import torch
from .protein import *
from .MAP import *
from .residue_constants import *
import warnings
warnings.filterwarnings("ignore")
import numpy as np
id2atom = {i:atom_type for i, atom_type in enumerate(atom_types)}
id2restype = {restype: i for i, restype in restype_order3.items()}

def CG_mapping(prot: Protein):
    p = [CoarseGrained.mapping[id2restype[aatype]] for aatype in prot.aatype]
    # Get the name, mass and coordinates for all atoms in the residue
    a = [[(id2atom[i], CoarseGrained.mass.get(id2atom[i][0], 0), CoarseGrained.electrons.get(id2atom[i][0], 0), prot.atom_positions[resid, i]) for i in np.where(atom_mask)[0]] for resid, atom_mask in enumerate(prot.atom_mask)]
    # Store weight, coordinate and index for atoms that match a bead
    q = [[[(m, coord, e, a[k].index((atom, m, e, coord))) for atom, m, e, coord in a[k] if atom in i] for i in r] for k, r in enumerate(p)]
    bead_sizes = {2:3.4, 3:4.1, 4: 4.7, 5:4.7}
    center_coords = []
    center_electrons = []
    center_sizes = []
    for resid, resi in enumerate(q):
        for bead in resi:
            mwx, es, ids = zip(*[((m*x, m*y, m*z), e, i) for m, (x, y, z), e, i in bead]) 
            mwx_ = np.array(mwx)
            tm  = sum([j[0] for j in bead]) 
            center_coord = mwx_.sum(axis=0)/tm
            center_coords.append(center_coord)
            center_electrons.append(sum(es))
            center_size = bead_sizes[len(bead)]
            center_sizes.append(center_size)
    return np.array(center_coords), np.array(center_electrons), np.array(center_sizes)