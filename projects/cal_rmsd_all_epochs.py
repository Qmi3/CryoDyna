import os
import sys
sys.path.insert(0,'/lustre/grp/gyqlab/lism/CryoDyna_CG')
import pickle
from cryostar.utils.align import get_rmsd, get_Ca_rmsd, get_backbone_rmsd
work_dir = '/lustre/grp/gyqlab/lism/CryoDyna/spike_MD_cg/atom_spike_MD_updated'
num_step = '0079_0124960'
save_dir = f'{work_dir}/{num_step}/sampled_pdb'
# work_dir = '/lustre/grp/gyqlab/lism/cryostar/projects/1ake_cg/atom_1ake_0/'
# num_step = '0030_0024000'
# save_dir = f'{work_dir}/{num_step}/sampled_pdb_BB2CA'
# work_dir = '/lustre/grp/gyqlab/lism/cryostar/projects/spike_MD_atom_HIS/atom_spike_MD_HIS'
# num_step = '0029_0011700'
# save_dir = f'{work_dir}/{num_step}/sampled_pdb'
rmsd_all = []
# rmsd_bb = []
rmsd_Ca = []
for i in range(0,100000,1000):
# for i in range(0,50000,100):
#     print(i, i//1000 + 1)
#     pdb_idx = i//1000 + 1
    # gt_path = f'/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/pdbs/1akeA_{pdb_idx}.pdb'
#     pdb_path = f'{save_dir}/sampled_pdb_{i}/FINAL/final_cg2at_de_novo_fixed.pdb'
    pdb_idx = i
    gt_path = f'/lustre/grp/gyqlab/share/cryoem_particles/spike_simulate/sampled_pdbs/frame_{pdb_idx}.pdb'
    pdb_path = f'{save_dir}/sampled_pdb_{i}/FINAL/final_cg2at_de_novo_fixed.pdb'
    print(gt_path, pdb_path)
    if not os.path.exists(pdb_path):
        print(f'{pdb_path} not found')
        continue
    rmsd_all.append(get_rmsd(gt_path, pdb_path))
    # rmsd_bb.append(get_backbone_rmsd(gt_path, pdb_path))
    rmsd_Ca.append(get_Ca_rmsd(gt_path, pdb_path))
pickle.dump(rmsd_all,open(f'{work_dir}/{num_step}/rmsd_all.pkl','wb'))
# pickle.dump(rmsd_bb,open(f'{work_dir}/{num_step}/rmsd_bb.pkl','wb'))
pickle.dump(rmsd_Ca,open(f'{work_dir}/{num_step}/rmsd_Ca.pkl','wb'))