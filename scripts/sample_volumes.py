import os
import argparse
import numpy as np
import torch
import sys
from mmengine import Config
from cryodyna.utils.dataio import StarfileDataSet, StarfileDatasetConfig
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from projects.train_density import CryoModel
from cryodyna.utils.mrc_tools import save_mrc
from cryodyna.utils.latent_space_utils import get_nearest_point, cluster_kmeans

def parse_args():
    parser = argparse.ArgumentParser(description='Sample density maps from latent space using CryoDyna')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='Working directory for atom model (e.g., 10792/atom_10792_10811)')
    parser.add_argument('--step_num', type=str, required=True,
                        help='Step number for atom model (e.g., 0049_0045650)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for MRC files (default: {work_dir}/{step_num}/sampled_vols)')
    parser.add_argument('--pixel_size', type=float, required=True,
                        help='Pixel size for MRC files ')
    # 修改：支持 npy 或 txt 文件
    parser.add_argument('--indices', type=str, default=None,
                        help='File path to indices (.npy or .txt) or comma-separated indices (e.g., "400,401,402")')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of K-means clusters (if indices not provided, cluster latent space with this K)')
    # parser.add_argument('--n_samples', type=int, default=20,
    #                     help='Number of samples to generate when using K-means (default: 20)')
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
    # density_dir = args.density_dir
    # density_step = args.density_step
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = f'{work_dir}/{step_num}/sampled_vols'
    else:
        output_dir = args.output_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载z向量
    z_path = f'{work_dir}/{step_num}/z.npy'
    print(f"Loading latent vectors from {z_path}")
    z = np.load(z_path)
    print(f"z shape: {z.shape}")
    
    # 确定要采样的中心（支持 npy 或 txt 文件）
    if args.indices is not None:
        # 使用从文件或字符串加载的索引
        centers_ids = load_indices(args.indices, z_list_length=len(z))
        print(f"Using {len(centers_ids)} indices from input")
        
    elif args.n_clusters is not None:
        # 对z进行K-means聚类
        print(f"Performing K-means clustering with K={args.n_clusters}...")
        # kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        # kmeans.fit(z)
        kmeans_labels, centers = cluster_kmeans(z, args.n_clusters)
        centers, centers_ids = get_nearest_point(z, centers)
        # # 获取聚类中心
        # cluster_centers = kmeans.cluster_centers_
        
        # # 找到距离每个聚类中心最近的z向量
        # from scipy.spatial.distance import cdist
        # distances = cdist(z, cluster_centers)
        # centers_ids = np.argmin(distances, axis=0)
        
        print(f"Found {len(centers_ids)} cluster centers")
        # centers_ids = centers_ids[:args.n_samples]  # 只取前n_samples个
        # print(f"Using first {len(centers_ids)} cluster centers for sampling")
        
    else:
        # 尝试加载之前保存的centers_ind.txt
        centers_ind_path = f'{work_dir}/{step_num}/centers_ind.txt'
        if os.path.exists(centers_ind_path):
            print(f"Loading indices from {centers_ind_path}")
            centers_ids = np.loadtxt(centers_ind_path).astype(int)
            print(f"Loaded {len(centers_ids)} indices")
        else:
            # 交互式输入
            print("No indices or n_clusters provided, and centers_ind.txt not found.")
            user_input = input("Please enter cluster indices (comma-separated) or number of K-means clusters: ")
            if ',' in user_input:
                centers_ids = np.array([int(x.strip()) for x in user_input.split(',')])
            else:
                k_val = int(user_input)
                print(f"Performing K-means clustering with K={k_val}...")
                kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                kmeans.fit(z)
                cluster_centers = kmeans.cluster_centers_
                from scipy.spatial.distance import cdist
                distances = cdist(z, cluster_centers)
                centers_ids = np.argmin(distances, axis=0)
                print(f"Found {len(centers_ids)} cluster centers")
    
    # 获取对应的z向量
    centers = z[centers_ids]
    centers = torch.from_numpy(centers).to(args.device)
    print(f"Sampling {len(centers)} density maps...")
    
    # 加载密度模型配置
    cfg_path = f'{work_dir}/config.py'
    print(f"Loading density model config from {cfg_path}")
    cfg = Config.fromfile(cfg_path)
    
    # 加载数据集
    print("Loading dataset...")
    dataset = StarfileDataSet(
        StarfileDatasetConfig(
            dataset_dir=cfg.dataset_attr.dataset_dir,
            starfile_path=cfg.dataset_attr.starfile_path,
            apix=cfg.dataset_attr.apix,
            side_shape=cfg.dataset_attr.side_shape,
            down_side_shape=cfg.data_process.down_side_shape,
            down_method=cfg.data_process.down_method,
            mask_rad=cfg.data_process.mask_rad,
            power_images=1.0,
            ignore_rots=False,
            ignore_trans=False, ))
    
    # 加载密度模型
    print("Loading density model...")
    cryo_model = CryoModel(cfg, dataset).to(args.device)
    weights_path = f'{work_dir}/{step_num}/ckpt.pt'
    print(f"Loading weights from {weights_path}")
    weights = torch.load(weights_path, map_location=args.device)
    cryo_model.vol.load_state_dict(weights)
    cryo_model.vol.eval()
    
    # 采样并保存密度图
    print(f"Generating and saving density maps to {output_dir}")
    pixel_size = args.pixel_size
    
    for i in range(len(centers)):
        # 生成密度图
        v = cryo_model.vol.make_volume(centers[i:i+1])
        
        # 保存MRC文件
        mrc_path = f"{output_dir}/centers_{i:02d}.mrc"
        save_mrc(v.cpu().numpy(), mrc_path, pixel_size,
                 -pixel_size * (v.shape[0] // 2))
        print(f"Saved {mrc_path}")
    
    # 保存使用的索引信息（同时保存 txt 和 npy 格式）
    indices_txt_path = f"{output_dir}/sampled_indices.txt"
    np.savetxt(indices_txt_path, centers_ids, fmt='%d')
    print(f"Saved indices to {indices_txt_path}")
    
    # indices_npy_path = f"{output_dir}/sampled_indices.npy"
    # np.save(indices_npy_path, centers_ids)
    # print(f"Saved indices as npy to {indices_npy_path}")
    
    # # 可选：保存对应的z向量
    # z_sampled_path = f"{output_dir}/sampled_z.npy"
    # np.save(z_sampled_path, centers.cpu().numpy())
    # print(f"Saved sampled z vectors to {z_sampled_path}")
    
    print(f"Done! Generated {len(centers)} density maps in {output_dir}")

if __name__ == "__main__":
    main()