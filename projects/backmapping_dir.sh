work_dir='../1ake_cg/cg_1ake/0003_0003124/pca-1'
# work_dir="../1ake_cg/atom_1ake_1/0000_0000000/pca-1"
for i in `ls ${work_dir}/*.pdb`
do
    echo $i
    sbatch ./backmapping.sh $i
done
