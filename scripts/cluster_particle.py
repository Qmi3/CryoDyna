import numpy as np
import argparse
import os
import pickle
from cryodyna.utils.latent_space_utils import get_nearest_point, cluster_kmeans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_file', required=True, help='Input z.npy path')
    parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters')
    parser.add_argument('--output_dir', required=True, help='Output npy path')
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading {args.z_file}")
    z = np.load(args.z_file)
    
    # K-means聚类
    print(f"Clustering with K={args.n_clusters}...")
    kmeans_labels, centers = cluster_kmeans(z, args.n_clusters)
    # centers, centers_ids = get_nearest_point(z, centers)
    
    # 保存结果
    for cluster_id in range(args.n_clusters):
        class_indices = np.where(kmeans_labels == cluster_id)[0]
        
        output_file = os.path.join(args.output_dir, f'cluster_{cluster_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(class_indices, f)
        
        print(f"Cluster {cluster_id}: {len(class_indices)} indices -> {output_file}")

if __name__ == "__main__":
    main()