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
    
    # 修改：支持 npy 或 txt 文件
    parser.add_argument('--indices', type=str, default=None,
                        help='File path to indices (.npy or .txt) or comma-separated indices (e.g., "400,401,402")')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of K-means clusters (if indices not provided, cluster latent space with this K)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    return parser.parse_args()

def load_indices_from_file(file_path):
    """
    从 .npy 或 .txt 文件加载索引
    
    Args:
        file_path: 文件路径 (.npy 或 .txt)
    
    Returns:
        numpy array of indices
    """
    file_path = str(file_path)
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Indices file not found: {file_path}")
    
    # 根据扩展名选择加载方式
    if file_path.endswith('.npy'):
        print(f"Loading indices from npy file: {file_path}")
        indices = np.load(file_path)
        # 确保是一维数组
        indices = indices.flatten()
        
    elif file_path.endswith('.txt'):
        print(f"Loading indices from txt file: {file_path}")
        # 支持多种 txt 格式：每行一个数字，或者空格/逗号分隔
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # 尝试按行解析
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # 如果有多行，每行一个数字
        if len(lines) > 1:
            indices = np.array([int(line) for line in lines])
        else:
            # 单行，尝试按空格或逗号分隔
            # 替换逗号为空格，然后分割
            content = content.replace(',', ' ')
            numbers = content.split()
            if numbers:
                indices = np.array([int(num) for num in numbers])
            else:
                raise ValueError(f"Empty or invalid txt file: {file_path}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Please use .npy or .txt file")
    
    print(f"Loaded {len(indices)} indices from {file_path}")
    print(f"Index range: [{indices.min()}, {indices.max()}]")
    
    return indices

def load_indices(indices_arg, z_list_length=None):
    """
    加载索引，支持三种格式：
    1. .npy 文件路径
    2. .txt 文件路径
    3. 逗号分隔的索引字符串
    """
    if indices_arg is None:
        return None
    
    # 检查是否为文件路径 (.npy 或 .txt)
    if isinstance(indices_arg, str) and (indices_arg.endswith('.npy') or indices_arg.endswith('.txt')):
        indices = load_indices_from_file(indices_arg)
    else:
        # 尝试解析为逗号分隔的整数列表
        try:
            # 如果传入的是字符串，按逗号分隔
            if isinstance(indices_arg, str):
                indices = np.array([int(x.strip()) for x in indices_arg.split(',')])
            else:
                # 如果已经是列表或其他类型
                indices = np.array(indices_arg)
            print(f"Using comma-separated indices: {indices}")
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid indices format: {indices_arg}. "
                           f"Provide either a .npy file, .txt file, or comma-separated integers (e.g., '400,401,402')")
    
    # 验证索引有效性
    if z_list_length is not None:
        if len(indices) > 0:
            if np.max(indices) >= z_list_length:
                raise ValueError(f"Index {np.max(indices)} exceeds z_list length {z_list_length}")
            if np.min(indices) < 0:
                raise ValueError(f"Negative index found: {np.min(indices)}")
    
    return indices

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
    
    # 确定要解码的索引（支持 npy 或 txt 文件）
    if args.indices is not None:
        # 使用从文件或字符串加载的索引
        centers_ids = load_indices(args.indices, z_list_length=len(z_list))
        print(f"Using {len(centers_ids)} indices from input")
        
    elif args.n_clusters is not None:
        # 对z进行K-means聚类
        print(f"Performing K-means clustering with K={args.n_clusters}...")
        kmeans_labels, centers = cluster_kmeans(z_list, args.n_clusters)
        centers, centers_ids = get_nearest_point(z_list, centers)
        print(f"Found {len(centers_ids)} cluster centers")
        
    else:
        # 如果都没有指定，默认使用所有索引
        print(f"No indices or clusters specified. Using all {len(z_list)} indices...")
        centers_ids = np.arange(len(z_list))
    
    # 加载模型权重
    weights_path = f'{work_dir}/{step_num}/ckpt.pt'
    print(f"Loading model weights from {weights_path}")
    weights = torch.load(weights_path, map_location=args.device)
    model.load_state_dict(weights['model'], strict=False)
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

if __name__ == "__main__":
    main()