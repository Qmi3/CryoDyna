#!/bin/bash
#SBATCH -o 1ake.%j.out
#SBATCH -p gpu33,gpu31
#SBATCH --gres=gpu:1
#SBATCH -J 1ake
#SBATCH --nodes=1
#SBATCH -t 200:00:00
# python projects/train_atom_ori.py projects/atom_configs/1ake.py
# python projects/train_atom_ori.py projects/atom_configs/spike_MD.py
# python projects/train_atom.py projects/atom_configs/1ake.py
# python projects/train_atom_ori.py projects/atom_configs/10059.py
python projects/train_cg.py projects/cg_configs/1ake.py
# python projects/train_cg.py projects/cg_configs/spike_MD_updated.py