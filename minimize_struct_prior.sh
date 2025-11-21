#!/bin/bash
#SBATCH -o spike_cg.%j.out
#SBATCH -p gpu31,gpu33,gpu35
#SBATCH --qos=normal
#SBATCH -J spike_cg
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00

source /opt/intel/oneapi/setvars.sh

mkdir projects/struct_prior/1akeA_50/MIN

gmx_mpi editconf -f projects/struct_prior/1akeA_50/1akeA_50_cg.pdb -o projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb -d 1.0 -bt cubic

gmx_mpi grompp -f projects/struct_prior/em.mdp -p projects/struct_prior/1akeA_50/1akeA_50_cg.top -c projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb  -o projects/struct_prior/1akeA_50/MIN/min -maxwarn 1

gmx_mpi mdrun -v -deffnm projects/struct_prior/1akeA_50/MIN/min -c projects/struct_prior/1akeA_50/MIN/min.pdb
