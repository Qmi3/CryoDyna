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
export PATH=/share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin:$PATH
# source /opt/intel/oneapi/setvars.sh
# /share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin/gmx
i=$1
output=`echo $i | sed 's/\.[^.]*$//'`
echo $output
echo ${output}_box.pdb

# Replace “gmx_mpi” with your path to gmx / gmx_mpi!!!
gmx editconf -f $i -o ${output}_box.pdb -d 1.0 -bt cubic

# Change the executable file path if required.
python ~/CG2AT2-Backward/database/bin/cg2at_backward.py -c ${output}_box.pdb -w tip3p -fg martini_2-2_charmm36_Jul2021 -ff charmm36-jul2021 -loc $output -silent -rna_ff charmm36

echo -e "1\n1\n0" | gmx trjconv -f $output/MERGED/checked_ringed_lipid_de_novo.pdb -s $output/MERGED/NVT/merged_cg2at_de_novo_nvt.tpr -o $output/FINAL/final_cg2at_de_novo_fixed.pdb -pbc cluster -center

rm ${output}_box.pdb