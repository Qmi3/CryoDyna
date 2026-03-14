import os
import torch
import numpy as np
import pickle
import sys
sys.path.insert(0,'/lustre/grp/gyqlab/lism/CryoDyna')
from cryodyna.utils.align import get_rmsd_loss
# from cryostar.utils.polymer import Polymer
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings("ignore")
# work_dir = "/lustre/grp/gyqlab/lism/CryoDyna/spike_MD_cg/atom_spike_MD_updated/"
work_dir = sys.argv[1]
# work_dir = '/lustre/grp/gyqlab/lism/cryostar/projects/spike_MD_HIS_cg_all_morse0.1/atom_spike_MD_HIS_0'
# num_step = '0079_0031200'

parser = PDBParser(QUIET=True)
for num_step in os.listdir(work_dir):
    if num_step == '0000_0000000':
        continue
    print(num_step)
    if not os.path.exists(f'{work_dir}/{num_step}/sampled_pdb'):
        continue
    print(num_step)
    rmsd_ori = []
    # for i in range(50000):
    #     print(i, i//1000 + 1)
    #     pdb_idx = i//1000 + 1
    for i in range(0, 100000, 100):
        # gt_path = f'/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/pdbs_cg/1akeA_{pdb_idx}.pdb'
        gt_path = f'/lustre/grp/gyqlab/share/cryoem_particles/spike_simulate/spike_pdbs_cg/frame_{i}_cg.pdb'
        save_dir = f'{work_dir}/{num_step}/sampled_pdb'
        pred_path = f'{save_dir}/sampled_pdb_{i}.pdb'
        print(gt_path, pred_path)
        gt_struct = parser.get_structure(id='none', file=gt_path)
        pred_struct = parser.get_structure(id='none', file=pred_path)
        pred = [j.coord for j in pred_struct.get_atoms()]
        pred = torch.tensor(np.array(pred, dtype=float)).unsqueeze(0)
        gt = [j.coord for j in gt_struct.get_atoms()]
        gt = torch.tensor(np.array(gt, dtype=float)).unsqueeze(0)
        rmsd_aln = get_rmsd_loss(gt, pred)
        rmsd = rmsd_aln['rmsd']
        rmsd_ori.append(float(rmsd.detach().cpu().numpy()))
    pickle.dump(rmsd_ori,open(f'{work_dir}/{num_step}/rmsd_cg_gt.pkl','wb'))