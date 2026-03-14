#!/bin/bash
#SBATCH -o cg2at.%j.out
#SBATCH -p cpu_short
#SBATCH --qos=normal
#SBATCH -J cg2at
#SBATCH -n 256
#SBATCH -t 1:00:00

work_dir='/lustre/grp/gyqlab/lism/CryoDyna/spike_MD_snr001_cg/atom_spike_MD_updated/0079_0124960/sampled_pdb'
for i in `ls ${work_dir}/*.pdb`
do
    echo $i
    sbatch ./slurm_cg2at $i
done
/lustre/grp/gyqlab/lism/CryoDyna/spike_MD_snr001_cg/atom_spike_MD_updated/0079_0124960/sampled_pdb/sampled_pdb_28000/FINAL/final_cg2at_de_novo_fixed.pdb