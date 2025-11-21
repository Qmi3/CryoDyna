#!/bin/bash
#SBATCH -o spike_cg.%j.out
#SBATCH -p gpu31,gpu33,gpu35
#SBATCH --qos=normal
#SBATCH -J spike_cg
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00

cryodyna/martini/martinize-na.py -f projects/struct_prior/1akeA_50/1akeA_50.pdb \
-o projects/struct_prior/1akeA_50/1akeA_50_cg.top \
-x projects/struct_prior/1akeA_50/1akeA_50_cg.pdb \
-sep \
-dssp /lustre/grp/gyqlab/lism/anaconda3/envs/cryostar/bin/mkdssp # replace with the path to the mkdssp executable 

