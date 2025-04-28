dataset_attr = dict(
    dataset_dir="/lustre/grp/gyqlab/share/cryoem_particles/IgG-RL/images/snr0.01",
    starfile_path="/lustre/grp/gyqlab/share/cryoem_particles/IgG-RL/images/snr0.01/snr0.01.star",
    apix=3.0,
    side_shape=128,
    ref_pdb_path="/lustre/grp/gyqlab/share/cryoem_particles/IgG-RL/pdbs/IgG_000_centered.pdb",
)
work_dir_name = "IgG-RL"
extra_input_data_attr = dict(
    nma_path="",
    use_domain=False,
    domain_path=None,
    ckpt_path=None,
    pred_sequence='/lustre/grp/gyqlab/share/cryoem_particles/IgG-RL/pdbs/IgG_af.fasta',
    protein_feature_path='/lustre/grp/gyqlab/zhangcw/alphafold/IgG/features.pkl',
    fold_feature_path='/lustre/grp/gyqlab/zhangcw/alphafold/IgG/result_model_1_multimer_v3_pred_1.pkl',
    # jax_param_path = '/lustre/grp/gyqlab/share/AF2_database/params/params_model_1_multimer_v3.npz'
)

data_process = dict(
    down_side_shape=None,
    mask_rad=1.0,
    # optional
    low_pass_bandwidth=7.68,
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
knn_num = 32
gmm = dict(tunable=False)

# model = dict(model_type="VAE",
#              input_space="real",
#              ctf="v2",
#              model_cfg=dict(
#                 encoder_cls='GAT',
#                 decoder_cls='MLP',
#                 e_hidden_dim=(512, 256, 128, 64, 32),
#                 #  e_hidden_dim=(512, 256, 128),
#                 #  z_dim=128,
#                 z_dim=8,
#                 latent_dim = 32,
#                 attention_layer = 2,
#                 #  d_hidden_dim=(128,128),
#                 #  d_e_hidden_dim=32,
#                 d_hidden_dim=(512, 256, 128, 64, 32)[::-1],
#                 # #  d_hidden_dim=(512,32,12)[::-1],
#                  e_hidden_layers=5,
#                 d_hidden_layers=5,
#              ))
model = dict(model_type="VAE",
             input_space="real",
             ctf="v2",
             model_cfg=dict(
                 encoder_cls='MS-GAT',
                 decoder_cls='metaGNN',
                 e_hidden_dim=(512, 256, 128, 128),
                 z_dim=128,
                 latent_dim = 32,
                 attention_layer = 2,
                 d_hidden_dim=(128,128),
                #  d_e_hidden_dim=32,
                #  d_hidden_dim=(512, 256, 128, 64, 32)[::-1],
                #  d_hidden_dim=(512,32,12)[::-1],
                 e_hidden_layers=4,
                d_hidden_layers=2,
                #  pe_dim = 8,
             ))
loss = dict(
    intra_chain_cutoff=12.,
    inter_chain_cutoff=0.,
    intra_chain_res_bound=None,
    clash_min_cutoff=4.0,
    mask_rad_for_image_loss=0.9375,
    gmm_cryoem_weight=1.0,
    sse_weight=0.0,
    connect_weight=1.0,
    clash_weight=1.0, 
    dist_weight=1.0,
    dist_keep_ratio=0.99,
    af_keep_ratio=0.99,
    # plddt_weight=0.1,
    # violation_weight=1.0,
    warmup_step=10000,
    kl_beta_upper=0.5,
    free_bits=3.0,
    gmm_cryoem_all_ipa=False,
    )

optimizer = dict(lr=1e-4,)

analyze = dict(cluster_k=10, skip_umap=False)

runner = dict(log_every_n_step=100,)

trainer = dict(max_epochs=50,
               devices=1,
               precision="16-mixed",
            #    num_sanity_val_steps=0,
            #    val_check_interval=5)
               check_val_every_n_epoch=5)

pose_searcher = False
num_ipa = 1
