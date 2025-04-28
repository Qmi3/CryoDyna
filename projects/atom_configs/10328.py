dataset_attr = dict(
    dataset_dir="/lustre/grp/gyqlab/share/cryoem_particles/10328/data",
    starfile_path="/lustre/grp/gyqlab/share/cryoem_particles/10328/data/J505_stack.star",
    apix=1.059,
    side_shape=256,
    ref_pdb_path="/lustre/grp/gyqlab/share/cryoem_particles/10328/data/7k65_centered.pdb",
)

extra_input_data_attr = dict(
    nma_path="",
    use_domain=False,
    domain_path=None,
    ckpt_path=None,
    pred_sequence='../../openfold/7k65/fasta_dir/7k65.fasta',
    protein_feature_path='/lustre/grp/gyqlab/share/cryoem_particles/10328/data/predictions/7K65_1-7K65_2_model_1_multimer_v3_feature_dict.pkl',
    fold_feature_path='/lustre/grp/gyqlab/share/cryoem_particles/10328/data/predictions/7K65_1-7K65_2_model_1_multimer_v3_output_dict.pkl',
    # jax_param_path = '/lustre/grp/gyqlab/share/AF2_database/params/params_model_1_multimer_v3.npz'
)

data_process = dict(
    down_side_shape=128,
    mask_rad=1.0,
    # optional generally setting
    low_pass_bandwidth=10,
)

data_loader = dict(
    train_batch_per_gpu=64,
    val_batch_per_gpu=128,
    workers_per_gpu=8
    )

seed = 1
exp_name = ""
eval_mode = False
do_ref_init = True
work_dir_name = './work_dir_10328'
knn_num=32
gmm = dict(tunable=False)

model = dict(model_type="VAE",
             input_space="real",
             ctf="v2",
             model_cfg=dict(
                 encoder_cls='MLP',
                 decoder_cls='GNN_with_prior',
                 e_hidden_dim=(512, 256, 128, 64, 16),
                 latent_dim=8,
                 d_hidden_dim=32,
                #  d_hidden_dim=(12, 12, 16)[::-1],
                 e_hidden_layers=5,
                 d_hidden_layers=3,
                #  pe_dim = 8,
             ))

loss = dict(
    intra_chain_cutoff=12.,
    inter_chain_cutoff=0.,
    intra_chain_res_bound=None,
    nt_intra_chain_cutoff=15.,
    nt_inter_chain_cutoff=15.,
    nt_intra_chain_res_bound=None,
    clash_min_cutoff=4.0,
    mask_rad_for_image_loss=0.9375,
    gmm_cryoem_weight=1.0,
    connect_weight=1.0,
    sse_weight=0.0,
    dist_weight=0.1,
    # dist_penalty_weight=1.0,
    dist_keep_ratio=0.99,
    af_keep_ratio=0.95,
    clash_weight=1.0,
    warmup_step=10000,
    kl_beta_upper=0.5,
    free_bits=3.0)

optimizer = dict(lr=1e-4, )

analyze = dict(cluster_k=10, skip_umap=False, downsample_shape=112)

runner = dict(log_every_n_step=100)

trainer = dict(max_epochs=30,
               devices=2,
               precision="16-mixed",
               num_sanity_val_steps=0,
            #    val_check_interval=1000,
               check_val_every_n_epoch=1)

# pose_searcher = dict(base_healpy=1,
#                      kmin=12,
#                      t_extent=5,
#                      t_ngrid=7,
#                      niter=5, 
#                      nkeptposes=8,
#                      loss_fn="cor", 
#                      t_xshift=0,
#                      t_yshift=0,
#                      device="cuda",
#                     )
