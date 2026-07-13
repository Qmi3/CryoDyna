#!/bin/bash

# Please add the GROMACS executable directory to your PATH environment variable.
export PATH=/share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin:$PATH

i=$1
output=`echo $i | sed 's/\.[^.]*$//'`
echo $output
echo ${output}_box.pdb

# Replace “gmx” with your path to gmx / gmx_mpi!!!
gmx editconf -f $i -o ${output}_box.pdb -d 1.0 -bt cubic

# Change the executable file path if required.
python ~/CG2AT2-Backward/database/bin/cg2at_backward.py -c ${output}_box.pdb -w tip3p -fg martini_2-2_charmm36_Jul2021 -ff charmm36-jul2021 -loc $output -silent -rna_ff charmm36

echo -e "1\n1\n0" | gmx trjconv -f $output/MERGED/checked_ringed_lipid_de_novo.pdb -s $output/MERGED/NVT/merged_cg2at_de_novo_nvt.tpr -o $output/FINAL/final_cg2at_de_novo_fixed.pdb -pbc cluster -center

rm ${output}_box.pdb