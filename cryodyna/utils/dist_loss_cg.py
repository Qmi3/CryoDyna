import numpy as np
from scipy.spatial import distance

import torch
from torch import nn
import torch.linalg as LA


def calc_dist_by_pair_indices(coord_arr, pair_indices):
    coord_pair_arr = coord_arr[pair_indices]  # num_pair, 2, 3
    dist = np.linalg.norm(np.diff(coord_pair_arr, axis=1), ord=2, axis=-1)
    return dist.flatten()


def find_bonded_pairs(topology, num_beads):
    pairs = []
    bond_ref_lengths = []
    bond_kbs = []
    for i, top in enumerate(topology):
        begin = sum(num_beads[:i])
        for bond_type in ['BB', 'SC', 'Elastic short', 'Elastic long']:
            bonds = [i for i in top.bonds[bond_type]]
            if bonds:
                pairs.extend([[begin+i-1, begin+j-1] for i, j in [i.atoms for i in bonds]])
                bond_ref_lengths.extend([i.parameters[-2] for i in bonds]) #nm
                bond_kbs.extend([i.parameters[-1] for i in bonds])
        bonds = [i for i in top.bonds['Constraint']]
        if bonds:
            pairs.extend([[begin+i-1, begin+j-1] for i, j in [i.atoms for i in bonds]])
            bond_ref_lengths.extend([i.parameters[0] for i in bonds])
            bond_kbs.extend([100000 for i in bonds])
        bonds = [i for i in top.bonds['BB']]
    pairs = np.vstack(pairs)
    bond_ref_lengths = np.array(bond_ref_lengths)*10 # A
    bond_kbs = np.array(bond_kbs)/100/12.5 # kJ/mol/A^2  
    return pairs, bond_ref_lengths, bond_kbs

def find_angle_pairs(topology, num_beads):
    pairs = []
    angle_refs = []
    angle_kbs = []
    for c, top in enumerate(topology):
        begin = sum(num_beads[:c])
        for angle_type in ['BBB', 'BBS', 'SC']:
            angles = [i for i in top.angles[angle_type]]
            if angles:
                pairs.extend([[begin+i-1, begin+j-1, begin+k-1] for i, j, k in [i.atoms for i in angles]])
                angle_refs.extend([i.parameters[-2] for i in angles])
                angle_kbs.extend([i.parameters[-1] for i in angles])
    pairs = np.vstack(pairs)
    angle_refs = np.array(angle_refs)/180*np.pi
    angle_kbs = np.array(angle_kbs)/12.5 
    return pairs, angle_refs, angle_kbs

def find_torsion_pairs(topology, num_beads):
    pairs = []
    torsion_refs = []
    torsion_kbs = []
    torsion_muls = []
    for c, top in enumerate(topology):
        begin = sum(num_beads[:c])
        torsions = [i for i in top.dihedrals['BBBB'] if i.parameters and len(i.parameters) > 2]
        if torsions:
            pairs.extend([[begin+i-1, begin+j-1, begin+k-1, begin+l-1] for i, j, k, l in [i.atoms for i in torsions]])
            torsion_refs.extend([i.parameters[0] for i in torsions])
            torsion_kbs.extend([i.parameters[1] for i in torsions])
            torsion_muls.extend([i.parameters[2] for i in torsions])
    pairs = np.vstack(pairs)
    torsion_refs = np.array(torsion_refs)/180*np.pi
    torsion_kbs = np.array(torsion_kbs)/12.5 
    torsion_muls = np.array(torsion_muls)
    return pairs, torsion_refs, torsion_kbs, torsion_muls

def find_improper_torsion_pairs(topology, num_beads):
    pairs = []
    improper_refs = []
    improper_kbs = []
    for i, top in enumerate(topology):
        begin = sum(num_beads[:i])
        for torsion_type in ['BBBB', 'SC']:
            torsions = [i for i in top.dihedrals[torsion_type] if i.parameters and len(i.parameters) == 2]
            if torsions:
                pairs.extend([[begin+i-1, begin+j-1, begin+k-1, begin+l-1] for i, j, k, l in [i.atoms for i in torsions]])
                improper_refs.extend([i.parameters[0] for i in torsions])
                improper_kbs.extend([i.parameters[1] for i in torsions])

        torsions = [i for i in top.dihedrals['BSC'] if i.parameters]
        if torsions:
            pairs.extend([[begin+i-1, begin+j-1, begin+k-1, begin+l-1] for i, j, k, l in [i.atoms for i in torsions]])
            improper_refs.extend([i.parameters[0] for i in torsions])
            improper_kbs.extend([i.parameters[1] for i in torsions])

    pairs = np.vstack(pairs)
    improper_refs = np.array(improper_refs)/180*np.pi
    improper_kbs = np.array(improper_kbs)/12.5 
    return pairs, improper_refs, improper_kbs
        
def find_nonbonded_pairs(topology, 
                         num_beads, 
                         coord_arr,
                         chain_id_arr,
                         nonbonded_cutoff):
    chain_id_unique = np.array(list(dict.fromkeys(chain_id_arr)))
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    sel_mask = dist_map <= nonbonded_cutoff
    sel_mask = np.triu(sel_mask, k=1)
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    epsilons = []
    sigmas = []
    forcefield = topology[0].options['ForceField']
    for i,j in indices_in_pdb:
        chain_id_i = chain_id_arr[i]
        chain_id_j = chain_id_arr[j]
        top_id_i = np.where(chain_id_unique == chain_id_i)[0][0]
        top_id_j = np.where(chain_id_unique == chain_id_j)[0][0]
        atom_id_i = i - sum(num_beads[:top_id_i])
        atom_id_j = j - sum(num_beads[:top_id_j])
        bead_type_i = topology[top_id_i].atoms[int(atom_id_i)][1]
        bead_type_j = topology[top_id_j].atoms[int(atom_id_j)][1]
        bead_type_id_i = np.where(forcefield.bead_types == bead_type_i)[0][0]
        bead_type_id_j = np.where(forcefield.bead_types == bead_type_j)[0][0]
        epsilons.append(forcefield.eps_matrix[bead_type_id_i, bead_type_id_j])
        sigmas.append(forcefield.s_matrix[bead_type_id_i, bead_type_id_j]*10) # A   
    return np.vstack(indices_in_pdb), np.array(epsilons)/12.5, np.array(sigmas)

def find_quaint_cutoff_pairs(coord_arr,
                             chain_id_arr,
                             res_id_arr,
                             intra_chain_cutoff=12.,
                             inter_chain_cutoff =12.,
                             intra_chain_res_bound=None):
    sel_indices = []
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    # 1. intra chain
    sel_mask = dist_map <= intra_chain_cutoff
    sel_mask = np.triu(sel_mask, k=1)
    # get indices of valid pairs
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    indices_in_pdb = indices_in_pdb[chain_id_arr[indices_in_pdb[:, 0]] == chain_id_arr[indices_in_pdb[:, 1]]]
    # filter by res_id
    if intra_chain_res_bound is not None:
        assert res_id_arr is not None
        res_ids = res_id_arr[indices_in_pdb]
        res_id_dist = np.abs(np.diff(res_ids, axis=1)).flatten()
        indices_in_pdb = indices_in_pdb[res_id_dist <= intra_chain_res_bound]

    sel_indices.append(indices_in_pdb)

    # 2. inter chain
    if inter_chain_cutoff is not None:
        sel_mask = dist_map <= inter_chain_cutoff
        sel_mask = np.triu(sel_mask, k=1)
        indices_in_pdb = np.nonzero(sel_mask)
        indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
        indices_in_pdb = indices_in_pdb[chain_id_arr[indices_in_pdb[:, 0]] != chain_id_arr[indices_in_pdb[:, 1]]]
        sel_indices.append(indices_in_pdb)

    sel_indices = np.vstack(sel_indices)
    return sel_indices

def find_quaint_cutoff_pairs_common(coord_arr1,
                                    coord_arr2,
                                    cutoff=12.,):
    dist_map = distance.cdist(coord_arr1, coord_arr2, metric='euclidean')
    sel_mask = dist_map <= cutoff
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    return indices_in_pdb

def find_range_cutoff_pairs(coord_arr, min_cutoff=4., max_cutoff=10.):
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    sel_mask = (dist_map <= max_cutoff) & (dist_map >= min_cutoff)
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    return indices_in_pdb

def remove_duplicate_pairs(pairs_a, pairs_b, remove_flip=True):
    """Remove pair b from a"""
    s = max(pairs_a.max(), pairs_b.max()) + 1
    mask = np.zeros((s, s), dtype=bool)

    np.put(mask, np.ravel_multi_index(pairs_a.T, mask.shape), True)
    np.put(mask, np.ravel_multi_index(pairs_b.T, mask.shape), False)
    if remove_flip:
        np.put(mask, np.ravel_multi_index(np.flip(pairs_b, 1).T, mask.shape), False)
    return np.column_stack(np.nonzero(mask))

def remove_duplicate_pairs_dist(
    pairs_a, pairs_b, angle_pairs, torsion_pairs,
    remove_flip=True
):
    """Remove bonded/angle/torsion pairs from Morse pair list."""

    s = max(pairs_a.max(), pairs_b.max(), angle_pairs.max(), torsion_pairs.max()) + 1
    mask = np.zeros((s, s), dtype=bool)

    np.put(mask, np.ravel_multi_index(pairs_a.T, mask.shape), True)

    np.put(mask, np.ravel_multi_index(pairs_b.T, mask.shape), False)
    if remove_flip:
        np.put(mask, np.ravel_multi_index(np.flip(pairs_b, 1).T, mask.shape), False)

    angle_pairs_ij = angle_pairs[:, [0, 1]]
    angle_pairs_jk = angle_pairs[:, [1, 2]]
    angle_pairs_ik = angle_pairs[:, [0, 2]]
    angle_all = np.concatenate([angle_pairs_ij, angle_pairs_jk, angle_pairs_ik], axis=0)
    np.put(mask, np.ravel_multi_index(angle_all.T, mask.shape), False)
    if remove_flip:
        np.put(mask, np.ravel_multi_index(np.flip(angle_all, 1).T, mask.shape), False)

    torsion_pairs_ij = torsion_pairs[:, [0, 1]]
    torsion_pairs_jk = torsion_pairs[:, [1, 2]]
    torsion_pairs_kl = torsion_pairs[:, [2, 3]]
    torsion_pairs_ik = torsion_pairs[:, [0, 2]]
    torsion_pairs_il = torsion_pairs[:, [0, 3]]
    torsion_pairs_jl = torsion_pairs[:, [1, 3]]
    torsion_all = np.concatenate([torsion_pairs_ij, torsion_pairs_jk, torsion_pairs_kl, torsion_pairs_ik, torsion_pairs_il, torsion_pairs_jl], axis=0)
    np.put(mask, np.ravel_multi_index(torsion_all.T, mask.shape), False)
    if remove_flip:
        np.put(mask, np.ravel_multi_index(np.flip(torsion_all, 1).T, mask.shape), False)

    keep_mask = mask[tuple(pairs_a.T)]
    return pairs_a[keep_mask]

def remove_duplicate_pairs_Morse(
    pairs_a, epsilons, sigmas, pairs_b,
    remove_flip=True
):
    """Remove bonded pairs from Morse pair list."""

    s = max(pairs_a.max(), pairs_b.max()) + 1
    mask = np.zeros((s, s), dtype=bool)

    # Step 1: mark all pairs_a as True
    np.put(mask, np.ravel_multi_index(pairs_a.T, mask.shape), True)

    # Step 2: remove bonded pairs_b
    np.put(mask, np.ravel_multi_index(pairs_b.T, mask.shape), False)
    if remove_flip:
        np.put(mask, np.ravel_multi_index(np.flip(pairs_b, 1).T, mask.shape), False)

    # Step 5: filter the pairs
    keep_mask = mask[tuple(pairs_a.T)]
    return (
        pairs_a[keep_mask],
        epsilons[keep_mask],
        sigmas[keep_mask]
    )


def filter_same_chain_pairs(pair_ids, chain_id_arr):
    chain_ids = chain_id_arr[pair_ids]

    same_chain_mask = chain_ids[:, 0] == chain_ids[:, 1]

    pair_mask = []

    for u in np.unique(chain_ids):
        tmp = np.logical_and(chain_ids[:, 0] == u, same_chain_mask)
        if np.any(tmp):
            pair_mask.append(tmp)

    if len(pair_mask) > 0:
        return np.row_stack(pair_mask)
    else:
        return Nonew

def filter_different_chain_pairs(pair_ids, chain_id_arr):
    chain_ids = chain_id_arr[pair_ids]

    different_chain_mask = chain_ids[:, 0] != chain_ids[:, 1]

    return different_chain_mask

def filter_BB_pairs(pair_ids, chain_id_arr, num_beads, topology):
    chain_id_unique = np.array(list(dict.fromkeys(chain_id_arr)))
    pair_mask = []
    for i,j in pair_ids:
        chain_id_i = chain_id_arr[i]
        chain_id_j = chain_id_arr[j]
        top_id_i = np.where(chain_id_unique == chain_id_i)[0][0]
        top_id_j = np.where(chain_id_unique == chain_id_j)[0][0]
        atom_id_i = i - sum(num_beads[:top_id_i])
        atom_id_j = j - sum(num_beads[:top_id_j])
        atom_name_i = topology[top_id_i].atoms[int(atom_id_i)][4]
        atom_name_j = topology[top_id_j].atoms[int(atom_id_j)][4]
        if atom_name_i in ['BB', "BB1"] and atom_name_j in ['BB', "BB1"]:
            pair_mask.append(True)
        else:
            pair_mask.append(False)
    return pair_ids[pair_mask]



class DistLoss(nn.Module):

    def __init__(self, pair_ids, gt_dists, reduction="mean"):
        super().__init__()
        self.reduction = reduction

        self.register_buffer("pair_ids", torch.from_numpy(pair_ids).long())
        self.register_buffer("gt_dists", torch.from_numpy(gt_dists).float())

    def calc_pair_dists(self, batch_struc):
        batch_dist = batch_struc[:, self.pair_ids]  # bsz, num_pair, 2, 3
        batch_dist = LA.vector_norm(torch.diff(batch_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
        return batch_dist

    def forward(self, batch_struc):
        batch_dist = self.calc_pair_dists(batch_struc)
        mse = torch.pow(batch_dist - self.gt_dists.unsqueeze(0), 2)
        if self.reduction is None:
            return mse
        elif self.reduction == "mean":
            return torch.mean(mse)
        else:
            raise NotImplementedError
            

class BondLoss(nn.Module):

    def __init__(self, pair_ids, gt_dists, normal_pairs,weights=None):
        super().__init__()

        self.register_buffer("pair_ids", pair_ids)
        self.register_buffer("gt_dists", gt_dists)
        self.register_buffer("weights", weights)
        self.normal_pairs = normal_pairs

    def calc_pair_dists(self, batch_struc):
        batch_pairs = batch_struc[:, self.pair_ids]  # bsz, num_pair, 2, 3
        batch_dist = LA.vector_norm(torch.diff(batch_pairs, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
        return batch_dist

    def forward(self, batch_struc):
        bs = batch_struc.shape[0]
        batch_dist = self.calc_pair_dists(batch_struc)
        Ebond = torch.pow(batch_dist - self.gt_dists.unsqueeze(0), 2)
        if self.weights is not None:
            Ebond = Ebond * self.weights
        return Ebond.sum() / (bs * self.normal_pairs)


class AngleLoss(nn.Module):

    def __init__(self, pair_ids, gt_angles, normal_pairs, weights=None):
        super().__init__()
        self.register_buffer("pair_ids", pair_ids)
        self.register_buffer("gt_angles", gt_angles)
        self.register_buffer("weights", weights)
        self.normal_pairs = normal_pairs
    
    def cal_pair_angles(self, batch_struc):
        batch_pairs = batch_struc[:, self.pair_ids]
        vector1 = batch_pairs[:, :, 0] - batch_pairs[:, :, 1]
        vector2 = batch_pairs[:, :, 2] - batch_pairs[:, :, 1]
        vector1_norm = vector1 / vector1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        vector2_norm = vector2 / vector2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_thetas = (vector1_norm * vector2_norm).sum(dim=-1).clamp(-0.999999, 0.999999)
        batch_angles = torch.acos(cos_thetas) # [0, π]
        return batch_angles
    
    def forward(self, batch_struc):
        bs = batch_struc.shape[0]
        batch_angles = self.cal_pair_angles(batch_struc)
        Eangle = torch.pow(torch.cos(batch_angles) - torch.cos(self.gt_angles.unsqueeze(0)), 2)
        if self.weights is not None:
            Eangle = Eangle * self.weights
        return Eangle.sum() / (bs * self.normal_pairs)

class TorsionLoss(nn.Module):

    def __init__(self, pair_ids, gt_torsions, gt_torsions_mul,normal_pairs, weights=None):
        super().__init__()
        self.register_buffer("pair_ids", pair_ids)
        self.register_buffer("gt_torsions", gt_torsions)
        self.register_buffer("gt_torsions_mul", gt_torsions_mul)
        self.register_buffer("weights", weights)
        self.normal_pairs = normal_pairs
    
    def cal_pair_torsions(self, batch_struc):
        # 提取原子组
        batch_pairs = batch_struc[:, self.pair_ids]
        b1 = batch_pairs[:, :, 1] - batch_pairs[:, :, 0]
        b2 = batch_pairs[:, :, 2] - batch_pairs[:, :, 1]
        b3 = batch_pairs[:, :, 3] - batch_pairs[:, :, 2]

        b2_norm = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # 投影向量到垂直于 b2 的平面
        v = b1 - (b1 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm
        w = b3 - (b3 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm

        v_norm = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w_norm = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        x = (v_norm * w_norm).sum(dim=-1).clamp(min=-1+1e-8, max=1-1e-8)
        y = torch.sum(torch.cross(b2_norm, v_norm, dim=-1) * w_norm, dim=-1)

        torsions = torch.atan2(y, x)  
        return torsions 

    
    def forward(self, batch_struc):
        bs = batch_struc.shape[0]
        batch_torsions = self.cal_pair_torsions(batch_struc)
        delta = (batch_torsions*self.gt_torsions_mul.unsqueeze(0) - self.gt_torsions.unsqueeze(0) + torch.pi) % (2 * torch.pi) - torch.pi
        Etorsion = (1+torch.cos(delta))
        if self.weights is not None:
            Etorsion = Etorsion * self.weights
        return 2*Etorsion.sum() / (bs * self.normal_pairs)

class ImproperTorsionLoss(nn.Module):

    def __init__(self, pair_ids, gt_torsions, normal_pairs, weights=None):
        super().__init__()
        self.register_buffer("pair_ids", pair_ids)
        self.register_buffer("gt_torsions", gt_torsions)
        self.register_buffer("weights", weights)
        self.normal_pairs = normal_pairs
    
    def cal_pair_torsions(self, batch_struc):
        batch_pairs = batch_struc[:, self.pair_ids]
        b1 = batch_pairs[:, :, 1] - batch_pairs[:, :, 0]
        b2 = batch_pairs[:, :, 2] - batch_pairs[:, :, 1]
        b3 = batch_pairs[:, :, 3] - batch_pairs[:, :, 2]

        b2_norm = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        v = b1 - (b1 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm
        w = b3 - (b3 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm

        v_norm = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w_norm = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        x = (v_norm * w_norm).sum(dim=-1).clamp(min=-1+1e-8, max=1-1e-8)
        y = torch.sum(torch.cross(b2_norm, v_norm, dim=-1) * w_norm, dim=-1)

        torsions = torch.atan2(y, x)  
        return torsions  # 单位为弧度
    
    def forward(self, batch_struc):
        bs = batch_struc.shape[0]
        batch_torsions = self.cal_pair_torsions(batch_struc)

        delta = (batch_torsions - self.gt_torsions.unsqueeze(0) + np.pi) % (2 * np.pi) - np.pi
        Etorsion = torch.pow(delta, 2)
        if self.weights is not None:
            Etorsion = Etorsion * self.weights
        return 2*Etorsion.sum() / (bs * self.normal_pairs)
        

class MorseLoss(nn.Module):

    def __init__(self, pair_ids, epsilons, sigmas, normal_pairs, a=1.0):
        super().__init__()
        self.register_buffer("epsilons", epsilons)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("pair_ids", pair_ids)
        self.normal_pairs = normal_pairs
        self.a = a
    
    def calc_pair_dists(self, batch_struc):
        batch_dist = batch_struc[:, self.pair_ids]  # bsz, num_pair, 2, 3
        batch_dist = LA.vector_norm(torch.diff(batch_dist, dim=-2), axis=-1).squeeze(-1) # bsz, num_pair
        return batch_dist

    def forward(self, batch_struc):
        bs = batch_struc.shape[0]
        batch_dist = self.calc_pair_dists(batch_struc) # A
        EMorse = self.epsilons*((1-torch.exp(-self.a*(batch_dist-1.1225*self.sigmas)))**2)
        return 2*EMorse.sum() / (bs * self.normal_pairs)
