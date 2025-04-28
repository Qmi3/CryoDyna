base_dir = "/lustre/grp/gyqlab/share/zcw_share"

dataset_attr = dict(
    dataset_dir=f"{base_dir}/tutorial_data_1ake/uniform_snr0-0001_ctf",
    starfile_path=f"{base_dir}/tutorial_data_1ake/uniform_snr0-0001_ctf/simulation.star",
    apix=1.0, # 格子长度1A
    side_shape=128,
    ref_pdb_path=f"{base_dir}/tutorial_data_1ake/pdbs/1akeA_50.pdb",
)

work_dir = "1ake_work_dir/grad_CA"

data_process = dict(
    down_side_shape=128,
    mask_rad=1.0,
    down_apix=1.0,
    # optional
    low_pass_bandwidth=4.,
)

data_loader = dict(
    val_batch_per_gpu=1,
    workers_per_gpu=1,
)

gmm = dict(tunable=False)

model = dict(
             ctf="v2",
            )
             
loss = dict(
    mask_rad_for_image_loss=0.9375,)
