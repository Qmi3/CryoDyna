dataset_attr = dict(
    dataset_dir="10073/data",
    starfile_path="10073/data/shiny_correctpaths_cleanedcorruptstacks.star",
    apix=1.4,
    side_shape=380,
    f_mu=None,
    f_std=None)

extra_input_data_attr = dict(
    given_z='/share/home/zhangcw/tutorial_10073/CryoDyna/10073_test/atom_10073_epoch80/0079_0173600/z.npy',
    ckpt_path=None,
)
work_dir_name='10073_test'
data_process = dict(
    down_side_shape=128,
    down_method="interp",
    mask_rad=1.0,
)

data_loader = dict(
    train_batch_per_gpu=32,
    val_batch_per_gpu=64,
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
