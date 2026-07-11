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
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cryodyna
module load gromacs
source /opt/intel/oneapi/setvars.sh

mkdir -p projects/struct_prior/1akeA_50/MIN

# Replace “gmx_mpi” with your path to gmx / gmx_mpi!!!
gmx_mpi editconf -f projects/struct_prior/1akeA_50/1akeA_50_cg.pdb -o projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb -d 1.0 -bt cubic

gmx_mpi grompp -f projects/struct_prior/em.mdp -p projects/struct_prior/1akeA_50/1akeA_50_cg.top -c projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb  -o projects/struct_prior/1akeA_50/MIN/min -maxwarn 1

gmx_mpi mdrun -v -deffnm projects/struct_
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

# python projects/train_density.py projects/density_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/density_test"
# python projects/train_atom_ori.py projects/atom_configs/1ake.py
# python projects/train_atom_ori.py projects/atom_configs/spike_MD.py
# python projects/train_atom.py projects/atom_configs/1ake.py
# python projects/train_atom_ori.py projects/atom_configs/10059.py
# python projects/train_cg.py projects/cg_configs/1ake.py
# python projects/train_cg.py projects/cg_configs/spike_MD_updated.py