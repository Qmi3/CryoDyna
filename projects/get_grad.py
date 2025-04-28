# 1. input consensus structures, images, poses
from mmengine import Config
import sys
sys.path.insert(0,"/lustre/grp/gyqlab/zhangcw/CryoDyna/")
import os
import pickle 
import numpy as np
import torch
import einops
from torch.utils.data import DataLoader
from cryostar.utils.transforms import SpatialGridTranslate
from cryostar.utils.ctf_utils import CTFRelion, CTFCryoDRGN
from cryostar.utils.misc import log_to_current
from cryostar.utils.polymer import Polymer
from cryostar.utils.losses import calc_cor_loss
from cryostar.utils.dataio import StarfileDataSet, StarfileDatasetConfig, Mask
from cryostar.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryostar.gmm.gmm import EMAN2Grid, batch_projection, Gaussian
from miscs import low_pass_mask2d, infer_ctf_params_from_config
import logging
logger = logging.getLogger(__name__)


class GetGrad:

    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.dataset = dataset
        self.device = device

        # Define GMM
        meta = Polymer.from_pdb(cfg.dataset_attr.ref_pdb_path)
        log_to_current(f"Load reference structure from {cfg.dataset_attr.ref_pdb_path}")

        # ref
        ref_centers = torch.from_numpy(meta.coord).float()
        ref_amps = torch.from_numpy(meta.num_electron).float()
        ref_sigmas = torch.ones_like(ref_amps)
        ref_sigmas.fill_(2.)
        log_to_current(f"1st GMM blob amplitude {ref_amps[0].item()}, sigma {ref_sigmas[0].item()}")

        num_pts = len(meta)
        log_to_current(f"Reference structure has {num_pts} atom coordinates")
        self.ref_centers = ref_centers.to(self.device)
        self.gmm_sigmas = ref_sigmas.to(self.device)
        self.gmm_amps = ref_amps.to(self.device)

        # low-pass filtering
        if hasattr(cfg.data_process, "low_pass_bandwidth"):
            log_to_current(f"Use low-pass filtering w/ {cfg.data_process.low_pass_bandwidth} A")
            lp_mask2d = low_pass_mask2d(cfg.data_process.down_side_shape, cfg.data_process.down_apix,
                                        cfg.data_process.low_pass_bandwidth)
            self.lp_mask2d = torch.from_numpy(lp_mask2d).float().to(self.device)
        else:
            self.lp_mask2d = None

        #
        self.mask = Mask(cfg.data_process.down_side_shape, rad=cfg.loss.mask_rad_for_image_loss)
        self.mask.mask = self.mask.mask.to(self.device)

        # for projection
        grid = EMAN2Grid(side_shape=cfg.data_process.down_side_shape, voxel_size=cfg.data_process.down_apix)
        self.grid = grid

        ctf_params = infer_ctf_params_from_config(cfg)
        if cfg.model.ctf == "v1":
            self.ctf = CTFRelion(**ctf_params, num_particles=len(dataset))
            log_to_current("We will deprecate `model.ctf=v1` in a future version, use `model.ctf=v2` instead.")
        elif cfg.model.ctf == "v2":
            self.ctf = CTFCryoDRGN(**ctf_params, num_particles=len(dataset))
        else:
            raise NotImplementedError
        log_to_current(ctf_params)

        # translate image helper
        self.translator = SpatialGridTranslate(D=cfg.data_process.down_side_shape, device=self.device)

        self.apix = self.cfg.data_process.down_apix

    def low_pass_images(self, images):
        f_images = primal_to_fourier_2d(images)
        f_images = f_images * self.lp_mask2d
        images = fourier_to_primal_2d(f_images).real
        return images

    def get_batch_pose(self, batch):
        rot_mats = batch["rotmat"]
        # yx order
        trans_mats = torch.concat((batch["shiftY"].unsqueeze(1), batch["shiftX"].unsqueeze(1)), dim=1)
        trans_mats /= self.apix
        return rot_mats, trans_mats
    
    def _shared_projection(self, pred_struc, rot_mats):
        pred_images = self.batch_projection(
            gauss=Gaussian(
                mus=pred_struc,
                sigmas=self.gmm_sigmas.unsqueeze(0),  # (b, num_centers)
                amplitudes=self.gmm_amps.unsqueeze(0)),
            rot_mats=rot_mats,
            line_grid=self.grid.line())
        pred_images = einops.rearrange(pred_images, 'b y x -> b 1 y x')
        return pred_images

    def _apply_ctf(self, batch, real_proj, freq_mask=None):
        f_proj = primal_to_fourier_2d(real_proj)
        f_proj = self._apply_ctf_f(batch, f_proj, freq_mask)
        # Note: here only use the real part
        proj = fourier_to_primal_2d(f_proj).real
        return proj

    def _apply_ctf_f(self, batch, f_proj, freq_mask=None):
        pred_ctf_params = {k: batch[k].to(self.device) for k in ('defocusU', 'defocusV', 'angleAstigmatism') if k in batch}
        f_proj = self.ctf(f_proj, ctf_params=pred_ctf_params, mode="gt", frequency_marcher=None)
        if freq_mask is not None:
            f_proj = f_proj * self.lp_mask2d
        return f_proj
    
    def batch_projection(self, gauss, rot_mats: torch.Tensor, line_grid) -> torch.Tensor:
        """A quick version of e2gmm projection.

        Parameters
        ----------
        gauss: (b/1, num_centers, 3) mus, (b/1, num_centers) sigmas and amplitudes
        rot_mats: (b, 3, 3)
        line_grid: (num_pixels, 3) coords, (nx, ) shape

        Returns
        -------
        proj: (b, y, x) projections
        """

        centers = einops.einsum(rot_mats, gauss.mus, "b c31 c32, b nc c32 -> b nc c31")

        sigmas = einops.rearrange(gauss.sigmas, 'b nc -> b 1 nc')
        sigmas = 2 * sigmas**2

        proj_x = einops.rearrange(line_grid.coords.to(self.device), "nx -> 1 nx 1") - einops.rearrange(centers[..., 0], "b nc -> b 1 nc")
        proj_x = torch.exp(-proj_x**2 / sigmas)

        proj_y = einops.rearrange(line_grid.coords.to(self.device), "ny -> 1 ny 1") - einops.rearrange(centers[..., 1], "b nc -> b 1 nc")
        proj_y = torch.exp(-proj_y**2 / sigmas)

        proj = einops.einsum(gauss.amplitudes, proj_x, proj_y, "b nc, b nx nc, b ny nc -> b nx ny")
        proj = einops.rearrange(proj, "b nx ny -> b ny nx")
        return proj

    def get_grad(self):
        data_generator = DataLoader(self.dataset,
                                batch_size=self.cfg.data_loader.val_batch_per_gpu,
                                shuffle=False,
                                drop_last=False,
                                num_workers=self.cfg.data_loader.workers_per_gpu)

        # 2. batch iteration
        os.makedirs(self.cfg.work_dir, exist_ok=True)
        with open(f'{self.cfg.work_dir}/allgrds.pkl', 'wb') as f_grads, open(f'{self.cfg.work_dir}/allloss.pkl', 'wb') as f_loss:
            for i, batch in enumerate(data_generator): 
                B = len(batch["proj"])
                gt_images = batch["proj"].to(self.device)
                idxes = batch["idx"]
                rot_mats, trans_mats = self.get_batch_pose(batch)
                rot_mats = rot_mats.to(self.device)
                trans_mats = trans_mats.to(self.device)

                ref_CA = self.ref_centers.repeat(B, 1, 1).requires_grad_(True)
                print(self.ref_centers[0])
                # get gmm projections
                pred_gmm_images = self._shared_projection(ref_CA, rot_mats)

                # apply ctf, low-pass
                pred_gmm_images = self._apply_ctf(batch, pred_gmm_images, self.lp_mask2d)

                if trans_mats is not None:
                    gt_images = self.translator.transform(einops.rearrange(gt_images, "B 1 NY NX -> B NY NX"),
                                                        einops.rearrange(trans_mats, "B C2 -> B 1 C2"))
                
                if self.lp_mask2d is not None:
                    lp_gt_images = self.low_pass_images(gt_images)
                else:
                    lp_gt_images = gt_images
                gmm_proj_loss = calc_cor_loss(pred_gmm_images, lp_gt_images, self.mask)
                log_to_current(f"Sample {i} | gmm_proj_loss: {gmm_proj_loss}")

                # 2.3 calculate gradient
                gmm_proj_loss.backward(retain_graph=True)
                gradients = ref_CA.grad.clone().detach()

                # 保存梯度和 FRC 到文件中
                pickle.dump(gradients.cpu().numpy(), f_grads)
                pickle.dump(gmm_proj_loss.clone().detach().cpu().numpy(), f_loss)
                
                ref_CA.grad.zero_()

        # 读取保存的梯度和 FRC
        allgrds = []
        allloss = []
        with open(f'{self.cfg.work_dir}/allgrds.pkl', 'rb') as f_grads, open(f'{self.cfg.work_dir}/allloss.pkl', 'rb') as f_loss:
            try:
                while True:
                    allgrds.append(pickle.load(f_grads))
                    allloss.append(pickle.load(f_loss))
            except EOFError:
                pass

        allgrds = np.array(allgrds)
        allloss = np.array(allloss)
        with open(f'{self.cfg.work_dir}/allgrds_comb.pkl', 'wb') as f:
            pickle.dump(allgrds, f)
        with open(f'{self.cfg.work_dir}/allloss_comb.pkl', 'wb') as f:
            pickle.dump(allloss, f)

def main():
    config_path = sys.argv[1]
    # set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    log_to_current("Use cuda {}".format(use_cuda))
    if not use_cuda:
        logger.warning("WARNING: No GPUs detected")
    cfg = Config.fromfile(config_path)
    # load dataset
    dataset = StarfileDataSet(
        StarfileDatasetConfig(
            dataset_dir=cfg.dataset_attr.dataset_dir,
            starfile_path=cfg.dataset_attr.starfile_path,
            apix=cfg.dataset_attr.apix,
            side_shape=cfg.dataset_attr.side_shape,
            down_side_shape=cfg.data_process.down_side_shape,
            mask_rad=cfg.data_process.mask_rad,
            power_images=1.0,
            ignore_rots=False,
            ignore_trans=False, ))
    if cfg.dataset_attr.apix is None:
        cfg.dataset_attr.apix = dataset.apix
    if cfg.dataset_attr.side_shape is None:
        cfg.dataset_attr.side_shape = dataset.side_shape
    if cfg.data_process.down_side_shape is None:
        if dataset.side_shape > 256:
            cfg.data_process.down_side_shape = 128
            dataset.down_side_shape = 128
        else:
            cfg.data_process.down_side_shape = dataset.down_side_shape
    cfg.data_process["down_apix"] = dataset.apix
    if dataset.down_side_shape != dataset.side_shape:
        cfg.data_process.down_apix = dataset.side_shape * dataset.apix / dataset.down_side_shape

    log_to_current(f"Load dataset from {dataset.cfg.dataset_dir}, power scaled by {dataset.cfg.power_images}")
    log_to_current(f"Total {len(dataset)} samples")
    log_to_current(f"The dataset side_shape: {dataset.side_shape}, apix: {dataset.apix}")
    log_to_current(f"Set down-sample side_shape {dataset.down_side_shape} with apix {cfg.data_process.down_apix}")

    get_grad = GetGrad(cfg, dataset, device)
    get_grad.get_grad()
if __name__ == "__main__":
    main()