#!/bin/bash
#SBATCH --job-name=test_smoke            # 作业名称
#SBATCH --output=%j.out             # 标准输出文件 (%j 会被替换为 Job ID)
#SBATCH --error=%j.err              # 错误输出文件
#SBATCH --time=01:00:00             # 运行时间上限 (HH:MM:SS)
#SBATCH --ntasks=1                  # 任务数 (通常为1)
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --mem=24G                    # 内存需求
#SBATCH --partition=gpu2     # 分区名称 (根据集群修改)
#SBATCH --gres=gpu:1
work_dir='1ake_cg/atom_1ake/0003_0003124/pca-1'
for i in `ls ${work_dir}/*.pdb`
do
    echo $i
    ./backmapping.sh $i
done
