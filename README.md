## CryoDyna: Multiscale end-to-end modeling of cryo-EM macromolecule dynamics with physics-aware neural network

## User Guide
The detailed user guide can be found at [here](https://www.notion.so/Making-reasonable-molecule-dance-video-with-CryoDyna-88b1c421ec0c825481a48159ddcf709d?source=copy_link). This comprehensive guide provides in-depth information about the topic at hand. Feel free to visit the link if you're seeking more knowledge or need extensive instructions regarding the topic. To furthur help user install and apply CryoDyna to their own systems, We also provide test report and logs for the whole CryoDyna workflows on the 1ake dataset and EMPIAR-10073 dataset wihch can be found in the [here](https://osf.io/6pbmk/). 

## Installation

- Create a conda environment: 
```bash
conda create -n cryodyna python=3.9 -y 
conda activate cryodyna
```
- Clone this repository and install the package (You should first check your CUDA drriver and look up the closet version at https://data.pyg.org/whl/ and replace the PyTorch Geometric wheels in requirements.txt accordingly): 
```bash
git clone https://github.com/Qmi3/CryoDyna.git
cd CryoDyna 
conda install -c conda-forge pdbfixer
conda install -c ostrokach dssp
pip install -r requirements.txt # Make sure to install the torch version that matches your CUDA driver version.
pip install -e .
```

## Quick start

### Preliminary

You may need to prepare the resources below before running `CryoDyna` (see user guide):

- **Consensus map** from homogenous reconstruction
- **STAR file** with ctf, pose and path of each particle.
- **PDB structure** docked into the consensus map.

**PDB requirements**
- The structure may contain **proteins and nucleic acids only**.
- Protein residues must belong to the **20 canonical amino acids**.
- **Non-standard residues, ligands, and other hetero atoms are currently not supported.**

### Example Dataset

We use a simulated dataset of the **one-dimensional conformational transition of *Escherichia coli* adenylate kinase** between its closed state (PDB: 1AKE) and open state (PDB: 4AKE) as an example (This case includes aligned PDB structure).

Download the dataset:

```bash
wget https://zenodo.org/records/17581921/files/tutorial_data_1ake.zip
```
and extract the zip file to the path `./`:
```bash
unzip tutorial_data_1ake.zip -d ./
```
### Smoke Test

Before applying CryoDyna to the test dataset, we first run a smoke test to verify that the software has been successfully installed. This test includes model initialization, inference, loss function computation, etc. Please enter the following three commands in the terminal, one after another. Specially, the train_density.py requires a latent code `z` as input, which can be obtained from the output of train_atom.py or train_cg.py. We prepared a pre-trained latent code in the `projects/for_smoke_test/z.npy` file after running train_cg.py for 30 epochs.

```bash
python projects/train_atom.py projects/atom_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/residue_test"

python projects/train_cg.py projects/cg_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/bead_test"

python projects/train_density.py projects/density_configs/1ake.py --cfg-options eval_mode=True work_dir_name="1ake/density_test"
```

Upon success,, the test will output "You have passed residue-level/bead-level/volume decoder test." at the end of logs. Additionally, result files with the step number 0000_0000000 will be generated in the following directories under : 1ake/residue_test/atom_1ake, 1ake/bead_test, and 1ake/density_test.


### Training
CryoDyna predicts conformational heterogeneity in two distinct level (residue-level and bead-level). Here's an illustration of its process:

#### Stage1 Training the residue-level deformation field
<p align="center">
  <img width="500" src="cryodyna/images/CryoDyna.png">
</p>

In this step, we generate an ensemble of molecule structures from the particles with Ca/P atom representing each residue. Note that the `pdb` file is used in this step and it should be docked into the concensus map! (see user guide)

```shell
python projects/train_atom.py projects/atom_configs/1ake.py
```

The outputs will be stored in the `1ake/atom_xxxxx` directory, and we perform evaluations every 12,000 steps. Within this directory, you'll observe sub-directories with the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
atom_xxxxx/
├── 0000_0000000/
├── ...
├── 0019_0015620/        # evaluation results
│  ├── ckpt.pt           # model parameters
│  ├── input_image.png   # visualization of input cryo-EM images
│  ├── pca-1.pdb         # sampled coarse-grained atomic structures along 1st PCA axis
│  ├── pca-2.pdb
│  ├── pca-3.pdb
│  ├── pred.pdb          # sampled structures at Kmeans cluster centers
│  ├── pred_gmm_image.png
│  └── z.npy             # the latent code of each particle
|                        # a matrix whose shape is num_of_particle x 8
├── yyyymmdd_hhmmss.log  # running logs
├── config.py            # a backup of the config file
├── struc_fea.pkl        # a backup of pre-computed feature
└── train_atom.py        # a backup of the training script
```

#### Stage2: Training the bead-level deformation field
<p align="center">
  <img width="500" src="cryodyna/images/CryoDyna-CG.png">
</p>

In this step, we generate an ensemble of molecule structures from the particles with 1-6 beads representing each residue.

During the training of CryoDyna-CG, the model requires a MARTINI coarse-grained structural prior.

You could set cfg.dataset_attr.ref_pdb_path to the path of your all-atom structure and set cfg.dataset_attr.ref_cg_pdb_path = None in the config file. CryoDyna-CG will automatically perform the coarse-graining. It's noted that experimentally resolved structures may miss side-chain atoms. If the initial structure you provided lacks side chains, you should use PDBfixer for side-chain heavy atom restoration, with the specific commands provided below. For 1ake dataset, you can skip this step.

```shell
#example, you need to update the path to the PDB file for which you want to add missing side-chain heavy atoms.
pdbfixer 1akeA_50.pdb --add-atoms heavy --output=1akeA_50_fixed.pdb
```
**(Optionally)**, you could also provide a MARTINI-coarse-grained structure that has already been energy-minimized, which can help the structural regularization converge more quickly during the early training stage. 

Using 1ake as an example: First, run ``` ./martinize_struct_prior.sh ```  to generate the coarse-grained mapping from the all-atom structure. If your structure lacks side chains, you could use the pdbfixer tool to add missing heavy atoms before running the martinize_struct_prior.sh script. The expected output can found in `projects/struct_prior/1akeA_50/MARTINI.tar.gz`. 
Then, run ```./minimize_struct_prior.sh``` to perform energy minimization. This step requires that the user has GROMACS installed. The GROMACS program must match the CUDA Deriver version. You should replace the GROMACS path in this script to your local path. The expected output can found in `projects/struct_prior/1akeA_50/MIN.tar.gz`. We provide a pre-minimized coarse-grained structure in the `projects/struct_prior/1akeA_50/min_ref.pdb` file, which can be used directly for training. You should set cfg.dataset_attr.ref_cg_pdb_path = 'projects/struct_prior/1akeA_50/min_ref.pdb' in the config file `projects/cg_configs/1ake.py`.

After that, run

```shell
python projects/train_cg.py projects/cg_configs/1ake.py # start from all-atom structure
```
or 

```shell
python projects/train_cg.py projects/cg_configs/1ake.py  --cfg-options dataset_attr.ref_cg_pdb_path='projects/struct_prior/1akeA_50/min_ref.pdb' # start from pre-minimized coarse-grained structure
```

The outputs will be stored in the `1ake_cg/cg_xxxxx` directory, and we perform evaluations every 12,000 steps. Within this directory, you'll observe sub-directories with the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
cg_xxxxx/
├── 0000_0000000/
├── ...
├── 0029_0023430/        # evaluation results
│  ├── ckpt.pt           # model parameters
│  ├── input_image.png   # visualization of input cryo-EM images
│  ├── pca-1.pdb         # sampled coarse-grained atomic structures along 1st PCA axis
│  ├── pca-2.pdb
│  ├── pca-3.pdb
│  ├── pred.pdb          # sampled structures at Kmeans cluster centers
│  ├── pred_gmm_image.png
|  ├── z_distribution.png
│  └── z.npy             # the latent code of each particle
|                        # a matrix whose shape is num_of_particle x 8
├── yyyymmdd_hhmmss.log  # running logs
├── config.py            # a backup of the config file
└── train_cg.py        # a backup of the training script
```

After generating the bead-level structure, you may use a backmapping method to obtain the full-atom structure.
In our work, we use [CG2AT2 + Backward](https://github.com/PepperLee-sm/CG2AT2-Backward.git) for the backmapping procedure.

### Backmapping Example

As an example, we perform backmapping for the **PC1 trajectory at epoch 3**.

First, split the 10 structures contained in `pca-1.pdb` into individual `.pdb` files by running:
```bash
python projects/split_pdb.py 1ake_cg/cg_1ake/0003_0003124/pca-1.pdb
```
The resulting single-structure .pdb files will be saved in `1ake_cg/cg_1ake/0003_0003124/pca-1`.

Next, assuming that CG2AT2_Backward has already been properly configured, run `projects/backmapping_dir.sh` to perform the backmapping. The script backmapping_dir.sh serves as a sbatch submission wrapper that executes backmapping.sh in order, the underlying backmapping processing script. This step converts the coarse-grained structures along the PC1 trajectory into atomistic structures.

```text
cg_xxxxx/
├── 0000_0000000/
├── ...
├── 0003_0003124/        # evaluation results
│  ├── pca-1
│  │  │── 1
│  │  │── 2
│  │  │  │── FINAL
│  │  │  │  │── final_cg2at_de_novo_fixed.pdb # All-atom structure for the 2nd frame of the PC1 trajectory
│  │  │  │── INPUT
│  │  │  │── MERGED
│  │  │  │── PROTEIN
│  │  │── ...
│  │  │── 10
│  ├── ...
│  ├── pca-1.pdb         # sampled coarse-grained atomic structures along 1st PCA axis
|
│  ├── ...
```

### Validation: Training the density generator

In step 1/2, the atom generator assigns a latent code `z` to each particle image. In this step, we will drop the encoder and directly use the latent code as a representation of a partcile. You can execute the subsequent command to initiate the training of a density generator.

```shell
# change the xxx/z.npy path to the output of the above command, if you skip the above step, you can use the pre-trained latent code in the `projects/for_smoke_test/z.npy` file. This is a latent embedding from 30th epoch of the bead-level deformation field training on 1ake dataset.
python projects/train_density.py projects/density_configs/1ake.py --cfg-options eval_mode=False extra_input_data_attr.given_z=projects/for_smoke_test/z.npy
```

Results will be saved to `1ake_density/density_xxxxx`, and each subdirectory has the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
density_xxxxx/
├── 0019_0086820/          # evaluation results
│  ├── ckpt.pt             # model parameters
│  ├── vol_pca_1_000.mrc   # density sampled along the PCA axis
│  ├── ...
│  ├── vol_pca_3_009.mrc
│	 ├── vol_kmeans_000.mrc  #density sampled by K-means clustering
│  ├── ...
│  ├── vol_kmeans_009.mrc
│  ├── z.npy
│  ├── z_kmeans.txt        # sampled z values by K-means clustering
│  ├── z_kmeans_ind.txt    # sampled z indices by K-means clustering
│  ├── z_pca_1.txt         # sampled z values along the PCA axis
│  ├── z_pca_2.txt
│  └── z_pca_3.txt
│  ├── z_pca_ind_1.txt     # sampled z indices values along the PCA axis
│  ├── z_pca_ind_2.txt
│  └── z_pca_ind_3.txt
├── yyyymmdd_hhmmss.log    # running logs
├── config.py              # a backup of the config file
└── train_density.py       # a backup of the training script
```


## Reference
You may cite this software by:
```bibtex
@misc{zhang2025cryodynamultiscaleendtoendmodeling,
      title={CryoDyna: Multiscale end-to-end modeling of cryo-EM macromolecule dynamics with physics-aware neural network}, 
      author={Chengwei Zhang and Shimian Li and Yihao Niu and Zhen Zhu and Sihao Yuan and Sirui Liu and Yi Qin Gao},
      year={2025},
      eprint={2510.16510},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2510.16510}, 
}
```
