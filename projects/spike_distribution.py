import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import MDAnalysis as mda
import os
import gc

# topology = "/lustre/grp/gyqlab/share/cryoem_particles/spike_simulate/seed_structure.pdb"
# trajectory = "/lustre/grp/gyqlab/share/cryoem_particles/spike_simulate/spike_sampled_pdbs.xtc"
# u = mda.Universe(topology, trajectory)
# # 获取距离时间序列
# def compute_distance_series(resid1, resid2, chainID1, chainID2):
#     try:
#         u = mda.Universe(topology, trajectory)
#         atom1 = u.select_atoms(f"chainID {chainID1} and resid {resid1} and name CA")
#         atom2 = u.select_atoms(f"chainID {chainID2} and resid {resid2} and name CA")
#         if len(atom1) == 0 or len(atom2) == 0:
#             raise ValueError("CA 原子未找到")
#         dist_series = []
#         for ts in u.trajectory:
#             dist = np.linalg.norm(atom1.positions[0] - atom2.positions[0])
#             dist_series.append(dist)

#         print(f"计算 {chainID1}-{resid1}-{chainID2}-{resid2} 的距离时间序列, 平均距离为 {np.mean(dist_series):.2f}", flush=True)
#         # 手动释放
#         u.trajectory.close()
#         del u, atom1, atom2
#         gc.collect()
#         return (resid1, resid2, chainID1, chainID2, np.array(dist_series))
#     except Exception as e:
#         print(f"计算 {chainID1}-{resid1}-{chainID2}-{resid2} 出错: {e}", flush=True)
#         return None

# dist1 = compute_distance_series(476, 901, 'A', 'C')
# dist2 = compute_distance_series(484, 559, 'A', 'C')
# dist1 = compute_distance_series(499, 116, 'A', 'B')
# dist2 = compute_distance_series(499, 406, 'A', 'B')
# fluctuations = np.load("/lustre/grp/gyqlab/lism/CryoDyna_CG/fluctuations_spike.npy", allow_pickle=True)
# top_20 = fluctuations[:len(fluctuations) // 5]
# top_series = [(r1, r2, chainID1, chainID2, dists) for r1, r2, chainID1, chainID2, dists, _ in top_20]

# triple1 = [(r1, r2, chainID1, chainID2, dist1,std) for r1, r2, chainID1, chainID2, dist1, std in top_20 if r1 == 476 and r2 == 901 and chainID1 == 'A' and chainID2 == 'C'][0]
# triple2 = [(r1, r2, chainID1, chainID2, dist2,std) for r1, r2, chainID1, chainID2, dist2, std in top_20 if r1 == 484 and r2 == 559 and chainID1 == 'A' and chainID2 == 'C'][0]

# save dist1 and dist2
# np.save('dist1_spike_A499B116_A499B406.npy', np.array(dist1[4]))
# np.save('dist2_spike_A499B116_A499B406.npy', np.array(dist2[4]))

dist1 = np.load('/lustre/grp/gyqlab/lism/CryoDyna_CG/dist1_spike_A499B116_A499B406.npy')
dist2 = np.load('/lustre/grp/gyqlab/lism/CryoDyna_CG/dist2_spike_A499B116_A499B406.npy')
# 设置高分辨率
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(8, 6))
plt.rcParams['axes.linewidth'] = 1  # 边框线宽
plt.rcParams['xtick.major.width'] = 1  # x轴主刻度线宽
plt.rcParams['ytick.major.width'] = 1  # y轴主刻度线宽
plt.rcParams['xtick.major.size'] = 2  # x轴主刻度长度
plt.rcParams['ytick.major.size'] = 2  # y轴主刻度长度

# 定义字体属性
axis_font = FontProperties(size=18, weight='bold')
legend_font = FontProperties(size=18, weight='bold')

# 计算2D直方图
H, xedges, yedges = np.histogram2d(np.array(dist1), np.array(dist2), bins=100, density=True)

# 转换为百分比
H_percent = H / H.sum() * 100

# 创建图形
plt.figure(figsize=(8, 6), dpi=300)
plt.imshow(
    H_percent.T, 
    origin='lower', 
    aspect='auto',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap='GnBu',
    interpolation='gaussian'  # 添加平滑效果
)

# 添加colorbar并设置为百分比格式
cbar = plt.colorbar(format='%.2f%%')
cbar.set_label('Probability (%)', fontsize=12)
# 设置坐标轴标签和标题
# plt.xlim(50, 110)
# plt.ylim(50, 150)
plt.xlabel(r'A499-B116($\AA$)', fontsize=12,fontproperties=axis_font)
plt.ylabel(r'A499-B406($\AA$)', fontsize=12,fontproperties=axis_font)
plt.title('Ground Truth Distribution', fontsize=16, fontproperties=axis_font)

# 调整布局
plt.tight_layout()
plt.savefig('spike_gt_new.pdf', dpi=300)

work_dir = '/lustre/grp/gyqlab/lism/CryoDyna/spike_MD_cg/atom_spike_MD_updated'
num_step = '0079_0124960'
save_dir = f'{work_dir}/{num_step}/sampled_pdb_all'
dist1_pred = []
dist2_pred = []
# for i in range(0,100000,100):
#     pdb_idx = i
#     pdb_path = f'{save_dir}/sampled_pdb_{i}/FINAL/final_cg2at_de_novo_fixed.pdb'
#     if not os.path.exists(pdb_path):
#         print(f'{pdb_path} not found')
#         continue
#     u = mda.Universe(pdb_path)
#     # 选择三个氨基酸的Cα原子 (假设是残基10,25,40)
#     atom1 = u.select_atoms(f"chainID A and resid 476 and name CA")
#     atom2 = u.select_atoms(f"chainID C and resid 901 and name CA")
#     atom3 = u.select_atoms(f"chainID A and resid 484 and name CA")
#     atom4 = u.select_atoms(f"chainID C and resid 559 and name CA")

#     dist1_pred.append(np.linalg.norm(atom1.positions - atom2.positions))
#     dist2_pred.append(np.linalg.norm(atom3.positions - atom4.positions))
for pdb in os.listdir(save_dir):
    if not pdb.endswith('.pdb'):
        continue
    pdb_path = f'{save_dir}/{pdb}'
    if not os.path.exists(pdb_path):
        print(f'{pdb_path} not found')
        continue
    u = mda.Universe(pdb_path)
    # 选择三个氨基酸的Cα原子 (假设是残基10,25,40)
    atom1 = u.select_atoms(f"chainID A and resid 499 and name BB")
    atom2 = u.select_atoms(f"chainID B and resid 116 and name BB")
    atom3 = u.select_atoms(f"chainID A and resid 499 and name BB")
    atom4 = u.select_atoms(f"chainID B and resid 406 and name BB")
    dist1_pred.append(np.linalg.norm(atom1.positions - atom2.positions))
    dist2_pred.append(np.linalg.norm(atom3.positions - atom4.positions))

# save dist1_pred and dist2_pred
np.save(f'{work_dir}/{num_step}/dist1_pred_spike_A499B116_A499B406.npy', np.array(dist1_pred))
np.save(f'{work_dir}/{num_step}/dist2_pred_spike_A499B116_A499B406.npy', np.array(dist2_pred))

# dist1_pred = np.load('dist1_pred.npy')
# dist2_pred = np.load('dist2_pred.npy')

# 计算2D直方图
H, xedges_pred, yedges_pred = np.histogram2d(np.array(dist1_pred), np.array(dist2_pred), bins=100, density=True)

# 转换为百分比
H_percent = H / H.sum() * 100

# 创建图形
plt.figure(figsize=(8, 6), dpi=300)
plt.imshow(
    H_percent.T, 
    origin='lower', 
    aspect='auto',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap='GnBu',
    interpolation='gaussian'  # 添加平滑效果
)

# 添加colorbar并设置为百分比格式
cbar = plt.colorbar(format='%.2f%%')
cbar.set_label('Probability (%)', fontsize=12)

# 设置坐标轴标签和标题
# plt.xlim(50, 110)
# plt.ylim(50, 150)
plt.xlabel(r'A499-B116($\AA$)', fontsize=12,fontproperties=axis_font)
plt.ylabel(r'A499-B406($\AA$)', fontsize=12,fontproperties=axis_font)
plt.title('Predicted Distribution', fontsize=16, fontproperties=axis_font)

# 调整布局
plt.tight_layout()
plt.savefig(f'{work_dir}/{num_step}/spike_cg_new.pdf', dpi=300)
    