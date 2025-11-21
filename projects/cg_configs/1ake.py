import shutil
dataset_attr = dict(
    dataset_dir="/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/uniform_snr0-0001_ctf",
    starfile_path="/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/uniform_snr0-0001_ctf/simulation.star",
    apix=1.0,
    side_shape=128,
    ref_pdb_path="/lustre/grp/gyqlab/lism/CryoDyna/projects/struct_prior/1akeA_50/1akeA_50.pdb",
    ref_cg_pdb_path="/lustre/grp/gyqlab/lism/CryoDyna/projects/struct_prior/1akeA_50/MIN/min.pdb",
)

cg_attr = dict(
    dssp=shutil.which("mkdssp"),
    ssc=0.5,
)
work_dir_name = "1ake_cg"
extra_input_data_attr = dict(
    nma_path="",
    use_domain=False,
    ckpt_path=None,
)

data_process = dict(
    down_side_shape=None,
    mask_rad=1.0,
    # optional
    low_pass_bandwidth=4,
)

data_loader = dict(
    train_ratio = 0.975,
    train_batch_per_gpu=64,
    val_batch_per_gpu=128,
    workers_per_gpu=8,
)

seed = 1
exp_name = ""
eval_mode = False
do_ref_init = True
knn_num = 16
gmm = dict(tunable=False)

model = dict(model_type="VAE",
             input_space="real",
             ctf="v2",
             model_cfg=dict(
                 encoder_cls='MS-GAT',
                 decoder_cls='metaGNN',
                 e_hidden_dim=(512, 256, 128, 128),
                 z_dim=64,
                 latent_dim = 32,
                 attention_layer = 2,
                 d_hidden_dim=(128,128),
                 e_hidden_layers=4,
                 d_hidden_layers=2,
                 is_CG=True,
             ))
loss = dict(
    nonbonded_cutoff=12,
    mask_rad_for_image_loss=0.9375,
    gmm_cryoem_weight=1.0,
    bonded_weight=1.0,
    angle_weight=1.0,
    torsion_weight=1.0,
    improper_weight=1.0,
    nonbonded_weight=1.0,
    Morse_a=1.0,
    intra_chain_cutoff=12.,
    inter_chain_cutoff=0.,
    intra_chain_res_bound=None,  
    dist_weight=1.0,
    dist_keep_ratio=0.8)

optimizer = dict(lr=1e-4,)

analyze = dict(cluster_k=10, skip_umap=True)

runner = dict(log_every_n_step=100,)

trainer = dict(max_epochs=80,
               devices=1,
               precision="32",
               check_val_every_n_epoch=2)
