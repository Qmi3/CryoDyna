#!/bin/bash
#SBATCH -o cg2at.%j.out
#SBATCH -p cpu_short,cpu1,cpu2
#SBATCH --qos=normal
#SBATCH -J cg2at
#SBATCH -n 1
#SBATCH -t 1:00:00

source /opt/intel/oneapi/setvars.sh
i=$1
output=`echo $i | sed 's/\.[^.]*$//'`
echo $output
echo ${output}_box.pdb

# Replace “gmx_mpi” with your path to gmx / gmx_mpi!!!
gmx_mpi editconf -f $i -o ${output}_box.pdb -d 1.0 -bt cubic

# Change the executable file path if required.
python /lustre/grp/gyqlab/lism/CG2AT2-Backward/database/bin/cg2at_backward.py -c ${output}_box.pdb -w tip3p -fg martini_2-2_charmm36_Jul2021 -ff charmm36-jul2021 -loc $output -silent -rna_ff charmm36

echo -e "1\n1\n0" | gmx_mpi trjconv -f $output/MERGED/checked_ringed_lipid_de_novo.pdb -s $output/MERGED/NVT/merged_cg2at_de_novo_nvt.tpr -o $output/FINAL/final_cg2at_de_novo_fixed.pdb -pbc cluster -center
