dataset_attr = dict(
    dataset_dir="/lustre/grp/gyqlab/share/cryoem_particles/10792/data/",
    starfile_path="/lustre/grp/gyqlab/share/cryoem_particles/10792/data/merge_new.star",
    apix=1.16,
    side_shape=288,
    f_mu=None,
    f_std=None
)

extra_input_data_attr = dict(
    given_z='/lustre/grp/gyqlab/zhangcw/CryoDyna/10792/atom_10792_10811/0049_0045650/z.npy',
    ckpt_path=None,
)
work_dir_name='10792'
data_process = dict(
    down_side_shape=288,
    down_method="interp",
    mask_rad=1.0,
)

data_loader = dict(
    train_batch_per_gpu=8,
    val_batch_per_gpu=16,
    workers_per_gpu=4,
)


exp_name = ""
eval_mode = False

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
