## CryoDyna: Multiscale end-to-end modeling of cryo-EM macromolecule dynamics with physics-aware neural network
[📚 User Guide](https://byte-research.gitbook.io/cryostar/)

## User Guide
The detailed user guide can be found at [here](https://byte-research.gitbook.io/cryostar/). This comprehensive guide provides in-depth information about the topic at hand. Feel free to visit the link if you're seeking more knowledge or need extensive instructions regarding the topic. 

## Installation

- Create a conda environment: `conda create -n cryodyna python=3.9 -y`
- Clone this repository and install the package: `git clone https://github.com/Qmi3/cryodyna.git && cd cryodyna && pip install .`

## Quick start

### Preliminary

You may need to prepare the resources below before running `CryoDyna`:

- a concensus map (along with each particle's pose)
- a pdb file (which has been docked into the concensus map)

### Training
CryoDyna predicts conformational heterogeneity in two distinct level(residue-level and bead-level). Here's an illustration of its process:

#### Training the residue-level deformation field
In this step, we generate an ensemble of molecule structures from the particles with Ca/P atom representing each residue. Note that the `pdb` file is used in this step and it should be docked into the concensus map!

```shell
cd projects
python train_atom.py atom_configs/1ake.py
```

The outputs will be stored in the `work_dirs/atom_xxxxx` directory, and we perform evaluations every 12,000 steps. Within this directory, you'll observe sub-directories with the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
atom_xxxxx/
├── 0000_0000000/
├── ...
├── 0112_0096000/        # evaluation results
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
└── train_atom.py        # a backup of the training script
```

#### Stage2: Training the bead-level deformation field
In this step, we generate an ensemble of molecule structures from the particles with 1-6 beads representing each residue.

```shell
cd projects
python train_atom.py cg_configs/1ake.py
```

The outputs will be stored in the `work_dirs/atom_xxxxx` directory, and we perform evaluations every 12,000 steps. Within this directory, you'll observe sub-directories with the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
atom_xxxxx/
├── 0000_0000000/
├── ...
├── 0112_0096000/        # evaluation results
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
└── train_atom.py        # a backup of the training script
```

### Validation: Training the density generator

In step 1/2, the atom generator assigns a latent code `z` to each particle image. In this step, we will drop the encoder and directly use the latent code as a representation of a partcile. You can execute the subsequent command to initiate the training of a density generator.

```shell
# change the xxx/z.npy path to the output of the above command
python train_density.py density_configs/1ake.py --cfg-options extra_input_data_attr.given_z=xxx/z.npy
```

Results will be saved to `work_dirs/density_xxxxx`, and each subdirectory has the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
density_xxxxx/
├── 0004_0014470/          # evaluation results
│  ├── ckpt.pt             # model parameters
│  ├── vol_pca_1_000.mrc   # density sampled along the PCA axis, named by vol_pca_pca-axis_serial-number.mrc
│  ├── ...
│  ├── vol_pca_3_009.mrc
│  ├── z.npy
│  ├── z_pca_1.txt         # sampled z values along the 1st PCA axis
│  ├── z_pca_2.txt
│  └── z_pca_3.txt
├── yyyymmdd_hhmmss.log    # running logs
├── config.py              # a backup of the config file
└── train_density.py       # a backup of the training script
```


## Reference
You may cite this software by:
```bibtex

```
