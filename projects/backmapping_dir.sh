#!/bin/bash
#SBATCH -o cg2at.%j.out
#SBATCH -p cpu_short
#SBATCH --qos=normal
#SBATCH -J cg2at
#SBATCH -n 256
#SBATCH -t 1:00:00

# work_dir='1ake_cg/atom_1ake/0003_0003124/pca-1'
work_dir="1ake_cg/atom_1ake_1/0000_0000000/pca-1"
for i in `ls ${work_dir}/*.pdb`
do
    echo $i
    ./backmapping.sh $i
done
