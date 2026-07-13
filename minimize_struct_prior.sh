# 
# source /opt/intel/oneapi/setvars.sh

mkdir -p projects/struct_prior/1akeA_50/MIN

# Replace “gmx_mpi” with your path to gmx / gmx_mpi!!!
/share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin/gmx editconf -f projects/struct_prior/1akeA_50/1akeA_50_cg.pdb -o projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb -d 1.0 -bt cubic

/share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin/gmx grompp -f projects/struct_prior/em.mdp -p projects/struct_prior/1akeA_50/1akeA_50_cg.top -c projects/struct_prior/1akeA_50/MIN/1akeA_50_cg_box.pdb  -o projects/struct_prior/1akeA_50/MIN/min -maxwarn 1

/share/home/chensh/apps/gromacs-2026.0/build-gromacs/bin/gmx mdrun -v -deffnm projects/struct_prior/1akeA_50/MIN/min -c projects/struct_prior/1akeA_50/MIN/min.pdb
