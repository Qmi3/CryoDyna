# python scripts/sampled_pdb.py --work_dir atom_1ake_0 --step_num 0029_0023430 --n_clusters 5 --output_path test/
import sys
import os
import argparse
from mmengine import Config
from cryodyna.utils.polymer import Polymer
import torch
import numpy as np
import pickle
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from projects.miscs import VAE
from cryodyna.utils.align import get_rmsd_loss
# sys.path.insert(0,'/lustre/grp/gyqlab/zhangcw/CryoDyna/')
from cryodyna.utils.dist_loss import find_continuous_pairs
from scipy.spatial.distance import cdist
from cryodyna.utils.latent_space_utils import get_nearest_point, cluster_kmeans, run_pca, get_pc_traj, run_umap
from cryodyna.utils.vis_utils import plot_z_dist, save_tensor_image
from cryodyna.utils.pdb_tools import bt_save_pdb, extract_sec_ids_merge_small_blocks, build_metagraph_knn_from_centroids, get_kplus_neighbor_metanodes
import biotite.structure as struc

def parse_args():
    parser = argparse.ArgumentParser(description='Decode latent representations to generate protein structures')
    parser.add_argument('--work_dir', type=str, required=True, 
                        help='Working directory (e.g., spike_MD/atom_spike_MD_179)')
    parser.add_argument('--step_num', type=str, required=True, 
                        help='Step number (e.g., 0079_0124960)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for the PDB file (default: {work_dir}/{step_num}/generated.pdb)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Specific indices of latent vectors to decode (e.g., --indices 400 401 402)')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of K-means clusters (if indices not provided, cluster latent space with this K)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置路径
    work_dir = args.work_dir
    step_num = args.step_num
    
    # 设置输出路径
    if args.output_path is None:
        output_path = f'{work_dir}/{step_num}/generated.pdb'
    else:
        output_path = args.output_path
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading configuration from {work_dir}/config.py")
    cfg = Config.fromfile(f'{work_dir}/config.py')
    
    print(f"Loading reference structure from {cfg.dataset_attr.ref_pdb_path}")
    meta = Polymer.from_pdb(cfg.dataset_attr.ref_pdb_path)
    ref_centers = torch.from_numpy(meta.coord).float()
    ref_amps = torch.from_numpy(meta.num_electron).float()
    ref_sigmas = torch.ones_like(ref_amps)
    num_pts = len(meta)
    
    # 构建元图
    print("Building metagraph...")
    sec_ids, _ = extract_sec_ids_merge_small_blocks(cfg.dataset_attr.ref_pdb_path, min_block_len=3)
    meta_edge_index, centroids = build_metagraph_knn_from_centroids(pos=meta.coord, sec_ids=sec_ids, k=cfg.knn_num)
    sec_ids = torch.from_numpy(sec_ids).long()
    meta_edge_index = meta_edge_index.long()
    centroid_distances = cdist(centroids, centroids)
    edge_dist = torch.from_numpy(centroid_distances[meta_edge_index[0], meta_edge_index[1]]).float()
    pe_vector = torch.from_numpy(np.array([centroids[sec_id] - ref_centers[i] for i, sec_id in enumerate(sec_ids)]))
    meta_2_node_edge, meta_2_node_vector = get_kplus_neighbor_metanodes(ref_centers, sec_ids, centroids)
    
    # 加载模型
    in_dim = cfg.data_process.down_side_shape ** 2

    model = VAE(in_dim=in_dim, sec_ids=sec_ids, meta_edge_index=meta_edge_index, 
                edge_dist=edge_dist, out_dim=num_pts * 3, pe_vector=pe_vector,
                meta_2_node_edge=meta_2_node_edge, meta_2_node_vector=meta_2_node_vector,
                **cfg.model.model_cfg)
    
    # 加载z
    z_path = f'{work_dir}/{step_num}/z.npy'
    print(f"Loading latent vectors from {z_path}")
    z_list = np.load(z_path)
    z = torch.from_numpy(z_list)
    
    # 确定要解码的索引
    if args.indices is not None:
        # 使用指定的索引
        centers_ids = np.array(args.indices)
        print(f"Using specified indices: {centers_ids}")
    elif args.n_clusters is not None:
        # 对z进行K-means聚类
        print(f"Performing K-means clustering with K={args.n_clusters}...")
        kmeans_labels, centers = cluster_kmeans(z_list, args.n_clusters)
        centers, centers_ids = get_nearest_point(z_list, centers)
        # kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        # kmeans.fit(z_list)
        # centers_ids = kmeans.cluster_centers_.shape[0]
        # # 找到距离每个聚类中心最近的z向量
        # distances = cdist(z_list, kmeans.cluster_centers_)
        # centers_ids = np.argmin(distances, axis=0)
        # 或者直接使用聚类中心作为解码输入
        # centers_emb = torch.from_numpy(kmeans.cluster_centers_)
        print(f"Found {len(centers_ids)} cluster centers")
        
    
    # 加载模型权重
    weights_path = f'{work_dir}/{step_num}/ckpt.pt'
    print(f"Loading model weights from {weights_path}")
    weights = torch.load(weights_path, map_location=args.device)
    # import pdb;pdb.set_trace()
    model.load_state_dict(weights['model'],strict=False)
    model.eval()
    model.to(args.device)
    
    # 解码
    print(f"Decoding {len(centers_ids)} structures...")
    sampled_z = z[centers_ids]
    device = args.device
    
    with torch.no_grad():
        delta = model.eval_z(sampled_z.to(torch.float32).to(device))
    pred_struc = delta.reshape(-1, num_pts, 3) + ref_centers[None, :, :].to(device)
    
    # 保存结果
    print(f"Saving structures to {output_path}")
    ref_atom_arr = meta.to_atom_arr().copy()
    atom_arrs = []
    
    for i in range(pred_struc.shape[0]):
        tmp_struc = pred_struc[i].cpu().numpy()
        tmp_atom_arr = ref_atom_arr.copy()
        tmp_atom_arr.coord = tmp_struc
        atom_arrs.append(tmp_atom_arr)
    
    bt_save_pdb(output_path, struc.stack(atom_arrs))
    print(f"Done! Saved {len(atom_arrs)} structures to {output_path}")
    
    # 额外保存索引信息
    indices_path = output_path.replace('.pdb', '_indices.txt')
    np.savetxt(indices_path, centers_ids, fmt='%d')
    print(f"Saved indices to {indices_path}")

if __name__ == "__main__":
    main()