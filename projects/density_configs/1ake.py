dataset_attr = dict(
    dataset_dir="/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/uniform_snr0-0001_ctf",
    starfile_path="/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/uniform_snr0-0001_ctf/simulation.star",
    apix=1.0,
    side_shape=128,
    f_mu=0.0,
    f_std=330.53455,
)

extra_input_data_attr = dict(
    given_z='/lustre/grp/gyqlab/share/cryoem_particles/tutorial_data_1ake/1ake_CryoDyna_CG_epoch29/z.npy',
    ckpt_path=None,
)

data_process = dict(
    down_side_shape=128,
    down_method="interp",
    mask_rad=1.0,
)

data_loader = dict(
    train_batch_per_gpu=16,
    val_batch_per_gpu=64,
    workers_per_gpu=4,
)

exp_name = ""
eval_mode = False
work_dir_name="1ake"
model = dict(shift_data=False,
             shift_method="interp",
             enc_space="fourier",
             ctf="v1",
             hidden=1024,
             z_dim=64,
             pe_type="gau2",
             net_type="cryodrgn",)

# loss type
loss = dict(
    loss_fn="fmsf",
    mask_rad_for_image_loss=1,
    free_bits=3.0
)

trainer = dict(
    max_epochs=20,
    devices=1,
    precision="16-mixed",
    num_sanity_val_steps=0,
    val_check_interval=None,
    check_val_every_n_epoch=5
)
