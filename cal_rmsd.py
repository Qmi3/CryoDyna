import sys
from mmengine import Config
from cryostar.utils.polymer import Polymer
import torch
import numpy as np
import pickle
from cryostar.utils.align import get_rmsd_loss
sys.path.insert(0,'/lustre/grp/gyqlab/zhangcw/CryoDyna_std/')
from cryostar.utils.dist_loss import  find_continuous_pairs
from scipy.spatial.distance import cdist
from cryostar.utils.pdb_tools import bt_save_pdb, extract_sec_ids_merge_small_blocks, build_metagraph_knn_from_centroids ,get_kplus_neighbor_metanodes
work_dir = '/lustre/grp/gyqlab/zhangcw/CryoDyna/spike_MD/atom_spike_MD_112'
# work_dir = '/lustre/grp/gyqlab/zhangcw/CryoDyna/IgG-RL/atom_IgG-RL_60'
# num_step = '0019_0031240'
# work_dir = '/lustre/grp/gyqlab/zhangcw/CryoDyna/1ake_3layer_with_af/atom_1ake_59'
num_step = '0079_0062480'
cfg = Config.fromfile(f'{work_dir}/config.py')
meta = Polymer.from_pdb(cfg.dataset_attr.ref_pdb_path)
ref_centers = torch.from_numpy(meta.coord).float()
ref_amps = torch.from_numpy(meta.num_electron).float()
ref_sigmas = torch.ones_like(ref_amps)
num_pts = len(meta)
sec_ids, _ = extract_sec_ids_merge_small_blocks(cfg.dataset_attr.ref_pdb_path ,min_block_len=3)
meta_edge_index, centroids = build_metagraph_knn_from_centroids( pos=meta.coord,sec_ids=sec_ids,k=32)
sec_ids = torch.from_numpy(sec_ids).long()
meta_edge_index = meta_edge_index.long()
centroid_distances = cdist(centroids, centroids)
edge_dist = torch.from_numpy(centroid_distances[meta_edge_index[0], meta_edge_index[1]]).float()
pe_vector = torch.from_numpy(np.array([centroids[sec_id] - ref_centers[i] for i,sec_id in enumerate(sec_ids)]))
meta_2_node_edge, meta_2_node_vector = get_kplus_neighbor_metanodes(ref_centers,sec_ids,centroids)# node_meta_dist = cdist(ref_centers,centroids)
# node_meta_idx = torch.tensor([torch.arange(ref_centers.shape[0]).repeat_interleave(4),torch.argsort(node_meta_dist)[:,:4].flatten()])
# distance_matrix = cdist(ref_centers, ref_centers) + np.eye(num_pts) * 1e3
# column = np.argsort(distance_matrix,axis=-1)[:,:cfg.knn_num]
# row = np.arange(num_pts).repeat(cfg.knn_num)
# edge = np.stack([row,column.flatten()],axis=-1)
# bond_feat = np.zeros((len(meta.chain_id),len(meta.chain_id)))
# connect_pairs = find_continuous_pairs(meta.chain_id, meta.res_id, meta.atom_name)
# bond_feat[np.array(connect_pairs)[:,0],np.array(connect_pairs)[:,1]] = 1
# bond_feat[np.array(connect_pairs)[:,1],np.array(connect_pairs)[:,0]] = 1
# edge_bond = bond_feat[edge[:,0],edge[:,1]]
in_dim = 256 ** 2
from projects.miscs import VAE
model = VAE(in_dim=in_dim, sec_ids = sec_ids, meta_edge_index=meta_edge_index, edge_dist=edge_dist,out_dim = num_pts * 3, pe_vector=pe_vector ,meta_2_node_edge = meta_2_node_edge,meta_2_node_vector=meta_2_node_vector,**cfg.model.model_cfg)
z_list = np.load(f'{work_dir}/{num_step}/z.npy')
z = torch.from_numpy(z_list)
device = 'cuda:0'
sampled_z_index = np.arange(0,100000,1000)
sampled_z = z[sampled_z_index]
# in_dim = 256 ** 2
weights = torch.load(f'{work_dir}/{num_step}/ckpt.pt')
model.load_state_dict(weights['model'])
model.eval()
model.to(device)
delta = []
for i in range(1):
    z_slice = sampled_z[i*100:i*100+100]
    with torch.no_grad():
        delta_ = model.eval_z(z_slice.to(torch.float32).to(device))
    torch.cuda.empty_cache()
    delta.append(delta_)
delta = torch.cat(delta,dim=0)
pred_struc = delta.reshape(-1,num_pts,3) + ref_centers[None,:,:].to(device)
rmsd_gnn_rg = []
for i in range(0,100000,1000):
    # gt_path = f'/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/pdbs/1akeA_{i+1}.pdb'
    # gt_path = f'/lustre/grp/gyqlab/share/cryoem_particles/IgG-RL/pdbs/clean_pdbs/{i:03d}.pdb'
    gt_path = f'/lustre/grp/gyqlab/share/cryoem_particles/spike_simulate/sampled_pdbs/frame_{i}.pdb'
    gt_pdb = Polymer.from_pdb(gt_path)
    gt = torch.from_numpy(gt_pdb.coord).float()
    rmsd_aln = get_rmsd_loss(gt.to(device), pred_struc[i//1000].unsqueeze(dim=0))
    rmsd = rmsd_aln['rmsd']
    rmsd_gnn_rg.extend(rmsd.detach().cpu().numpy())
torch.cuda.empty_cache()
pickle.dump(rmsd_gnn_rg,open(f'{work_dir}/{num_step}/rmsd_0.001.pkl','wb'))
