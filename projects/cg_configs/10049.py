import shutil
dataset_attr = dict(
    dataset_dir="/lustre/grp/gyqlab/share/cryoem_particles/10049/data",
    starfile_path="/lustre/grp/gyqlab/share/cryoem_particles/10049/data/cryosparc_P53_J26_006_particles.star",
    apix=1.23,
    side_shape=192,
    ref_pdb_path="/lustre/grp/gyqlab/share/cryoem_particles/10049/data/af3-10049-align-new_centered.pdb",
    ref_cg_pdb_path="/lustre/grp/gyqlab/lism/cryostar/martini_test/10049/MIN/min.pdb",
)

cg_attr = dict(
    dssp=shutil.which("mkdssp"),
    ssc=0.5,
)
work_dir_name = "10049_cg"

extra_input_data_attr = dict(
    nma_path="",
    use_domain=False,
    domain_path=None,
    ckpt_path=None
)

data_process = dict(
    down_side_shape= None,
    mask_rad=1.0,
    # optional
    low_pass_bandwidth=16.86,
)

data_loader = dict(
    train_batch_per_gpu=256,
    val_batch_per_gpu=128,
    workers_per_gpu=8,
)

seed = 1
exp_name = ""
eval_mode = False
do_ref_init = False

gmm = dict(tunable=False)
knn_num = 32
gmm = dict(tunable=False)
model = dict(model_type="VAE",
             input_space="real",
             ctf="v2",
             model_cfg=dict(
                encoder_cls='MS-GAT',
                decoder_cls='metaGNN',
                e_hidden_dim=(512, 256, 128, 128),
                z_dim=64,
                latent_dim=32,
                attention_layer=2,
                d_hidden_dim=(256, 512),
                e_hidden_layers=4,
                d_hidden_layers=2,
                is_CG=True,
             ))

loss = dict(
    Morse_a=1.0,
    angle_weight=1.0,
    bonded_weight=1.0,
    dist_aa_nt_weight=1.0,
    dist_keep_ratio=0.99,
    dist_keep_ratio_aa_nt=0.99,
    dist_weight=1.0,
    free_bits=3.0,
    gmm_cryoem_weight=1.0,
    improper_weight=1.0,
    inter_chain_cutoff=0.0,
    inter_chain_cutoff_aa_nt=12.0,
    intra_chain_cutoff=12.0,
    intra_chain_res_bound=None,
    mask_rad_for_image_loss=0.9375,
    nonbonded_cutoff=12,
    nonbonded_weight=1.0,
    nt_inter_chain_cutoff=15.0,
    nt_intra_chain_cutoff=15.0,
    nt_intra_chain_res_bound=None,
    torsion_weight=1.0,
    warmup_step=10000)

optimizer = dict(lr=1e-4,)

analyze = dict(cluster_k=10, skip_umap=False)

runner = dict(log_every_n_step=100,)

trainer = dict(max_epochs=80,
               devices=1,
               precision="32",
               check_val_every_n_epoch=2)
