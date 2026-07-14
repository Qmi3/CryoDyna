#!/bin/bash
#SBATCH --job-name=test_smoke           # 作业名称
#SBATCH --output=%j.out             # 标准输出文件 (%j 会被替换为 Job ID)
#SBATCH --error=%j.err              # 错误输出文件
#SBATCH --time=01:00:00             # 运行时间上限 (HH:MM:SS)
#SBATCH --ntasks=1                  # 任务数 (通常为1)
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --mem=24G                    # 内存需求
#SBATCH --partition=gpu3        # 分区名称 (根据集群修改)
#SBATCH --gres=gpu:1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cryodyna
# module load gromacs
# 
# source /opt/intel/oneapi/setvars.sh

# mkdir -p projects/struct_prior/1akeA_50/MIN

# # Replace “gmx_mpi” with your path to gmx / gmx_mpi!!!
# /share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin/gmx editconf -f projects/struct_prior/1akeA_50/1akeA_50_cg.pdb -o projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb -d 1.0 -bt cubic
# python scripts/sample_volumes.py --work_dir 10792/atom_10792_10811 --step_num 0049_0045650 --indices 94235 --pixel_size 4.15625
# /share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin/gmx grompp -f projects/struct_prior/em.mdp -p projects/struct_prior/1akeA_50/1akeA_50_cg.top -c projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb  -o projects/struct_prior/1akeA_50/MIN/min -maxwarn 1
# python scripts/sample_pdbs.py --work_dir 1ake/atom_1ake_0 --step_num 0025_0020306 --n_clusters 5 --output_path 1ake/atom_1ake_0/0025_0020306/kemans5.pdb

# /share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin/gmx mdrun -v -deffnm projects/struct_prior/1akeA_50/MIN/min -c projects/struct_prior/1akeA_50/MIN/min.pdb
# projects/backmapping_dir.sh 1ake_cg/cg_1ake/0003_0003124/pca-1/1.pdb
# python projects/train_density.py projects/density_configs/1ake.py --cfg-options extra_input_data_attr.given_z=projects/for_smoke_test/z.npy
# python projects/train_cg.py projects/cg_configs/1ake.py  --cfg-options dataset_attr.ref_cg_pdb_path='projects/struct_prior/1akeA_50/1akeA_50_cg.pdb'
# python projects/train_density.py projects/density_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/density_test"
# python projects/train_cg.py projects/cg_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/bead_test"
# python projects/train_atom.py projects/atom_configs/1ake.py
# export PYTHONPATH="~/miniconda3/:$PYTHONPATH"
# python projects/train_cg.py projects/cg_configs/1ake.py
# python projects/train_atom.py projects/atom_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/residue_test"
# python projects/train_cg.py projects/cg_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/bead_test"
# python /share/home/zhangcw/density_simulation.py -i /share/home/zhangcw/pca-1-1of10.pdb -o /share/home/zhangcw/pca-1-1of10.mrc -s 256 -a 2.078 
pdbfixer /share/home/zhangcw/5gan_centered_clean.pdb --add-atoms heavy --output=/share/home/zhangcw/5gan_centered_clean_fixed.pdb
# python projects/train_density.py projects/density_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/density_test"
# python projects/train_atom_ori.py projects/atom_configs/1ake.py
# python projects/train_atom_ori.py projects/atom_configs/spike_MD.py
# python projects/train_atom.py projects/atom_configs/1ake.py
# python projects/train_atom_ori.py projects/atom_configs/10059.py
# python projects/train_cg.py projects/cg_configs/1ake.py
# python projects/train_cg.py projects/cg_configs/spike_MD_updated.py