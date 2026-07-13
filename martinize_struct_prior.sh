# You should prepare a dir to store all-atom structure and then perform cryodyna/martini/martinize.py to generate the coarse-grained mapping. T
cp tutorial_data_1ake/pdbs/1akeA_50.pdb projects/struct_prior/1akeA_50/
python cryodyna/martini/martinize.py -f projects/struct_prior/1akeA_50/1akeA_50.pdb \
-o projects/struct_prior/1akeA_50/1akeA_50_cg.top \
-x projects/struct_prior/1akeA_50/1akeA_50_cg.pdb \
-sep