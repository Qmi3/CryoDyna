# Ref: https://github.com/ml-struct-bio/cryodrgn.git

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from . import lie_tools, shift_grid, so3_grid
from .fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryodyna.gmm.gmm import EMAN2Grid, Gaussian
from cryodyna.utils.transforms import SpatialGridTranslate
import einops
import functools
from mmengine import print_log



log_to_current = functools.partial(print_log, logger="current")

def rot_2d(angle: float, outD: int, device: torch.device) -> torch.Tensor:
    rot = torch.zeros((outD, outD), device=device)
    rot[0, 0] = np.cos(angle)
    rot[0, 1] = -np.sin(angle)
    rot[1, 0] = np.sin(angle)
    rot[1, 1] = np.cos(angle)
    return rot


def to_tensor(x: Union[np.ndarray, torch.Tensor, None]):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def interpolate(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    # print(f"Interpolating {img.shape} {coords.shape}")
    assert len(coords.shape) == 2
    assert coords.shape[-1] == 2
    grid = coords * 2  # careful here! grid_sample expects [-1,1] instead of [-0.5,0.5]
    grid = grid[None, None, ...].expand(img.shape[0], -1, -1, -1)

    res = (
        F.grid_sample(
            img.unsqueeze(1),
            grid,
        )
        .squeeze(2)
        .squeeze(1)
    )

    return res

def low_pass_mask2d(shape, apix=1., bandwidth=2):
    freq = np.fft.fftshift(np.fft.fftfreq(shape, apix))
    freq = freq**2
    freq = np.sqrt(freq[:, None] + freq[None, :])
    mask = np.asarray(freq < 1 / bandwidth, dtype=np.float32)
    return mask


FAST_INPLANE = False


class PoseSearch:
    """Pose search"""

    def __init__(
        self,
        kmin: int,
        kmax: int,
        gmm_sigmas, 
        gmm_amps, 
        ctf, 
        base_healpy: int = 2,
        t_extent: int = 5,
        t_ngrid: int = 7,
        niter: int = 5,
        nkeptposes: int = 8,
        loss_fn: str = "msf",
        t_xshift: int = 0,
        t_yshift: int = 0,
        down_side_shape: int=128, 
        down_apix: float=1., 
        device: str='cpu',
    ):
        self.device = device

        self.gmm_sigmas = gmm_sigmas.to(self.device)
        self.gmm_amps = gmm_amps.to(self.device)
        self.ctf = ctf
        
        self.base_healpy = base_healpy
        self.so3_base_quat = so3_grid.grid_SO3(base_healpy)
        self.base_quat = (
            so3_grid.s2_grid_SO3(base_healpy) if FAST_INPLANE else self.so3_base_quat
        )
        self.so3_base_rot = lie_tools.quaternions_to_SO3(
            to_tensor(self.so3_base_quat)
        )
        self.base_rot = lie_tools.quaternions_to_SO3(to_tensor(self.base_quat))
        self.nbase = len(self.base_quat)
        self.base_inplane = so3_grid.grid_s1(base_healpy)
        self.base_shifts = torch.tensor(
            shift_grid.base_shift_grid(
                base_healpy - 1, t_extent, t_ngrid, xshift=t_xshift, yshift=t_yshift
            )
        ).float()
        self.t_extent = t_extent
        self.t_ngrid = t_ngrid

        self.kmin = kmin
        self.kmax = kmax
        self.niter = niter
        self.nkeptposes = nkeptposes
        self.loss_fn = loss_fn
        self._so3_neighbor_cache = {}  # for memoization
        self._shift_neighbor_cache = {}  # for memoization
        
        self.down_side_shape = down_side_shape
        self.down_apix = down_apix
        
        self.grid = EMAN2Grid(side_shape=self.down_side_shape, voxel_size=self.down_apix)
        
        self.translator = SpatialGridTranslate(D=self.down_side_shape, device=self.device)
        

    def eval_grid(
        self,
        *,
        structures: torch.Tensor,
        images: torch.Tensor,
        rot: torch.Tensor,
        trans:torch.Tensor, 
        k: int,
        angles_inplane: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        structures: B x L x 3
        images: B x 1 x X xY
        rot: Q x 3 x 3 rotation matrics
        trans: T x 2 translation matrics
        k: radius of fourier components to evaluate
        """
        B, _, X, Y = images.shape
        if len(trans.shape) == 2:
            T, _ = trans.shape
            trans = trans.unsqueeze(0).expand(B, T, 2) 
        else:
            assert trans.shape[0] == B
            _, T, _ = trans.shape
        # images = images.unsqueeze(1).expand(B, T, 1, X, Y).reshape(B*T, 1, X, Y)
        # trans = trans.unsqueeze(0).expand(B, T, 2).reshape(B*T, 2)
        # images = self.translator.transform(einops.rearrange(images, "B 1 NY NX -> B NY NX"),einops.rearrange(trans, "B C2 -> B 1 C2")) # (B*T, 1, X, Y)
        
        images = self.translator.transform(einops.rearrange(images, "B 1 NY NX -> B NY NX"),trans) # (B, T, X, Y)
        images = einops.rearrange(images, "B T NY NX -> B T 1 NY NX").reshape(B*T, 1, X, Y)
        
        
        def compute_err(images, structures, rot):
            _, _, X, Y = images.shape
            B, L, _ = structures.shape
            adj_angles_inplane = None
            if angles_inplane is not None:
                # apply a random in-plane rotation from the set
                # to avoid artifacts due to grid alignment
                rand_a = angles_inplane[np.random.randint(len(angles_inplane))]
                rand_inplane_rot = rot_2d(rand_a, 3, rot.device)
                rot = rand_inplane_rot @ rot
                adj_angles_inplane = angles_inplane - rand_a

            # 对空域图片做旋转（预测）平移（gt），在Fourier空间滤波求loss
            BQ, _, _ = rot.shape
            
            Q = BQ//B
            structures = structures.unsqueeze(1).expand(B, Q, L, 3).reshape(BQ, L, 3) # TODO: check dim
            # rot = rot.unsqueeze(0).expand(B, Q, 3, 3).reshape(B*Q, 3, 3)
            y_gmm = self._shared_projection(structures, rot) # [B*Q, 1, 128, 128]
            idxes = self.idxes.unsqueeze(1).repeat(1, Q).flatten()
            ctf_params = {k:v.unsqueeze(1).repeat(1, Q, 1, 1).reshape(B*Q, 1, 1) for k, v in self.ctf_params.items()}
            # apply ctfeinops.rearrange(images, "B 1 NY NX -> B NY NX")
            y_gmm = self._apply_ctf(y_gmm, idxes, ctf_params)# [B*Q, 1, 128, 128]
            if adj_angles_inplane is not None:
                y_gmm = self.rotate_images(y_gmm, adj_angles_inplane) # [BQ*len(angles), 1, 128, 128]  空域(未滤波)
            
            y_f = primal_to_fourier_2d(y_gmm).real
            images_f = primal_to_fourier_2d(images).real
            mask = to_tensor(low_pass_mask2d(self.down_side_shape, self.down_apix, self.down_side_shape/k)).to(self.device)
            y_f *= mask
            images_f *= mask
            
            images_f = images_f.view(B, -1, 1, X*Y)
            y_f = y_f.view(B, 1, -1, X*Y)

            if self.loss_fn == "mse":
                err = (images_f - y_f).pow(2).sum(-1)  # BxTxQ
            elif self.loss_fn == "msf":
                B, T, _, Npix = images_f.shape
                Npix = images_f.shape[-1]
                dots = (
                    images_f.view(B, -1, Npix)
                    @ y_f.view(y_f.shape[0], -1, Npix).transpose(-1, -2)
                ).view(B, T, -1)
                norm = (y_f * y_f).sum(-1) / 2

                err = -dots + norm  # BxTxQ

                # err1 = -(images * y_hat).sum(-1) + (y_hat * y_hat).sum(-1) / 2   # BxTxQ
                # delta = (err1 - err).abs().max() / (err1 + err).mean() < 1e-2
            elif self.loss_fn == "cor":
                err = -(images_f * y_f).sum(-1) / y_f.std(-1)
            else:
                raise NotImplementedError(f"Unknown loss_fn: {self.loss_fn}")
            return err

        err = compute_err(images, structures, rot)
        return err  # BxTxQ

    def _shared_projection(self, pred_struc, rot_mats):
        pred_images = self.batch_projection(
            gauss=Gaussian(
                mus=pred_struc,
                sigmas=self.gmm_sigmas.unsqueeze(0),  # (b, num_centers)
                amplitudes=self.gmm_amps.unsqueeze(0)),
            rot_mats=rot_mats,
            coords=self.grid.line().coords.to(self.device))
        pred_images = einops.rearrange(pred_images, 'b y x -> b 1 y x')
        return pred_images
    
    def batch_projection(self, gauss: Gaussian, rot_mats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
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

        proj_x = einops.rearrange(coords, "nx -> 1 nx 1") - einops.rearrange(centers[..., 0], "b nc -> b 1 nc")
        proj_x = torch.exp(-proj_x**2 / sigmas)

        proj_y = einops.rearrange(coords, "ny -> 1 ny 1") - einops.rearrange(centers[..., 1], "b nc -> b 1 nc")
        proj_y = torch.exp(-proj_y**2 / sigmas)

        proj = einops.einsum(gauss.amplitudes, proj_x, proj_y, "b nc, b nx nc, b ny nc -> b nx ny")
        proj = einops.rearrange(proj, "b nx ny -> b ny nx")
        return proj
    
    def _apply_ctf(self, real_proj, idxes, ctf_params):
        f_proj = primal_to_fourier_2d(real_proj)
        f_proj = self._apply_ctf_f(f_proj, idxes, ctf_params)
        # Note: here only use the real part
        proj = fourier_to_primal_2d(f_proj).real
        return proj

    def _apply_ctf_f(self, f_proj, idxes, ctf_params):
        f_proj = self.ctf(f_proj, idxes, ctf_params=ctf_params, mode="gt", frequency_marcher=None)
        return f_proj
    
    def rotate_images(self, images: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
        B, _, _, _ = images.shape
        res = torch.zeros((B, len(angles), images.size(2), images.size(3)), device=self.device)
        
        for angle_idx, angle in enumerate(angles):
            theta = torch.tensor([[[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0]]], dtype=torch.float, device=self.device).repeat(B, 1, 1)
            grid = F.affine_grid(theta, images.size(), align_corners=True)
            rotated_images = F.grid_sample(images, grid, align_corners=True)
            res[:, angle_idx] = rotated_images.squeeze(1)

        return res.view(B*len(angles), 1, images.size(2), images.size(3))
    
    def get_neighbor_so3(self, quat: np.ndarray, s2i: int, s1i: int, res: int):
        """Memoization of so3_grid.get_neighbor."""
        key = (int(s2i), int(s1i), int(res))
        if key not in self._so3_neighbor_cache:
            self._so3_neighbor_cache[key] = so3_grid.get_neighbor(quat, s2i, s1i, res)
        # FIXME: will this cache get too big? maybe don't do it when res is too
        return self._so3_neighbor_cache[key]

    def get_neighbor_shift(self, x, y, res):
        """Memoization of shift_grid.get_neighbor."""
        key = (int(x), int(y), int(res))
        if key not in self._shift_neighbor_cache:
            self._shift_neighbor_cache[key] = shift_grid.get_neighbor(
                x, y, res - 1, self.t_extent, self.t_ngrid
            )
        # FIXME: will this cache get too big? maybe don't do it when res is too
        return self._shift_neighbor_cache[key]

    def subdivide(
        self, quat: np.ndarray, q_ind: np.ndarray, cur_res: int
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Subdivides poses for next resolution level

        Inputs:
            quat (N x 4 tensor): quaternions
            q_ind (N x 2 np.array): index of current S2xS1 grid
            cur_res (int): Current resolution level

        Returns:
            quat  (N x 8 x 4) np.array
            q_ind (N x 8 x 2) np.array
            rot   (N*8 x 3 x 3) tensor
        """
        N = quat.shape[0]

        assert len(quat.shape) == 2 and quat.shape == (N, 4), quat.shape
        assert len(q_ind.shape) == 2 and q_ind.shape == (N, 2), q_ind.shape

        # get neighboring SO3 elements at next resolution level -- todo: make this an array operation
        neighbors = [
            self.get_neighbor_so3(quat[i], q_ind[i][0], q_ind[i][1], cur_res)
            for i in range(len(quat))
        ]
        quat = np.array([x[0] for x in neighbors])  # Bx8x4
        q_ind = np.array([x[1] for x in neighbors])  # Bx8x2
        rot = lie_tools.quaternions_to_SO3(torch.from_numpy(quat).view(-1, 4))

        assert len(quat.shape) == 3 and quat.shape == (N, 8, 4), quat.shape
        assert len(q_ind.shape) == 3 and q_ind.shape == (N, 8, 2), q_ind.shape
        assert len(rot.shape) == 3 and rot.shape == (N * 8, 3, 3), rot.shape

        return quat, q_ind, rot

    def keep_matrix(self, loss: torch.Tensor, B: int, max_poses: int) -> torch.Tensor:
        """
        Inputs:
            loss (B, T, Q): tensor of losses for each translation and rotation.

        Returns:
            keep (3, B * max_poses): bool tensor of rotations to keep, along with the best translation for each
        """
        shape = loss.shape
        assert len(shape) == 3
        best_loss, best_trans_idx = loss.min(1) # 每一个rot中最好的trans（最低的loss）
        flat_loss = best_loss.view(B, -1)
        flat_idx = flat_loss.topk(max_poses, dim=-1, largest=False, sorted=True)[1] # 找最低loss中排名前k个rot
        # add the batch index in, to make it completely flat
        flat_idx += (
            torch.arange(B, device=self.device).unsqueeze(1) * flat_loss.shape[1]
        )
        flat_idx = flat_idx.view(-1)

        keep_idx = torch.empty(
            len(shape), B * max_poses, dtype=torch.long, device=self.device
        )
        keep_idx[0] = flat_idx // shape[2]
        keep_idx[2] = flat_idx % shape[2]
        keep_idx[1] = best_trans_idx[keep_idx[0], keep_idx[2]]
        return keep_idx

    def getk(self, iter_: int) -> int:
        k = self.kmin + int(iter_ / self.niter * (self.kmax - self.kmin))
        return min(k, self.D // 2)
        # return min(self.Lmin * 2 ** iter_, self.Lmax)

    def opt_theta_trans(
        self,
        batch, 
        structures: torch.Tensor,
        init_poses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        images = batch["proj"].to(self.device)
        idxes = batch["idx"].to(self.device)
        self.idxes = idxes
        pred_ctf_params = {k: batch[k].to(self.device) for k in ('defocusU', 'defocusV', 'angleAstigmatism') if k in batch}
        self.ctf_params = pred_ctf_params
        if init_poses:
            init_poses = init_poses.to(self.device)
        structures = structures.to(self.device)

        B, _, D, D = images.shape
        self.D = D
        
        loss = rot = None
        if init_poses is None:
            # base_rot = self.base_rot  # Q x 3 x 3
            base_rot = self.base_rot.expand(
                    B, *self.base_rot.shape
                ).reshape(-1, 3, 3, )  # BQ x 3 x 3
            # Compute the loss for all poses
            k = self.getk(0)
            loss = self.eval_grid(
                structures=structures,
                images=images,
                rot=base_rot.to(self.device),
                trans=self.base_shifts.to(self.device), 
                k=k,
                angles_inplane=self.base_inplane if FAST_INPLANE else None,
            )
            log_to_current(f"Pose Search 0 Best Loss:")
            for i in range(B):
                log_to_current(f"{loss[i].min()}")
            keepB, keepT, keepQ = self.keep_matrix(
                loss, B, self.nkeptposes
            ).cpu()  # B x -1
        else:
            # careful, overwrite the old batch index which is now invalid
            keepB = (
                torch.arange(B, device=init_poses.device)
                .unsqueeze(1)
                .repeat(1, self.nkeptposes)
                .view(-1)
            )
            keepT, keepQ = init_poses.reshape(-1, 2).t()

        new_init_poses = (
            torch.cat((keepT, keepQ), dim=-1)
            .view(2, B, self.nkeptposes)
            .permute(1, 2, 0)
        )

        quat = self.so3_base_quat[keepQ]
        q_ind = so3_grid.get_base_ind(keepQ, self.base_healpy)  # Np x 2
        trans = self.base_shifts[keepT]
        shifts = self.base_shifts.clone()
        for iter_ in range(1, self.niter + 1):
            self.ctf_params = {k:v[keepB] for k,v in self.ctf_params.items()}

            k = self.getk(iter_)
            quat, q_ind, rot = self.subdivide(quat, q_ind, iter_ + self.base_healpy - 1)
            shifts /= 2
            trans = trans.unsqueeze(1) + shifts.unsqueeze(0)  # FIXME: scale
            rot = rot.to(self.device)
            self.idxes = self.idxes[keepB]
            loss = self.eval_grid(
                structures=structures[keepB],
                images=images[keepB],
                rot=rot,
                trans=trans.to(self.device), 
                k=k
            )  # sum(NP), 8

            # nkeptposes = 1
            # nkeptposes = max(1, math.ceil(self.nkeptposes / 2 ** (iter_-1)))
            log_to_current(f"Pose Search {iter_} Best Loss:")
            for i in range(B):
                log_to_current(f"{loss[i].min()}")
            nkeptposes = self.nkeptposes if iter_ < self.niter else 1

            keepBN, keepT, keepQ = self.keep_matrix(
                loss, B, nkeptposes
            ).cpu()  # B x (self.Nkeptposes*32)
            keepB = keepBN * B // loss.shape[0]  # FIXME: expain
            assert (
                len(keepB) == B * nkeptposes
            ), f"{len(keepB)} != {B} x {nkeptposes} at iter {iter_}"
            quat = quat[keepBN, keepQ]
            q_ind = q_ind[keepBN, keepQ]
            trans = trans[keepBN, keepT]

        assert loss is not None
        bestBN, bestT, bestQ = self.keep_matrix(loss, B, 1).cpu()
        assert len(bestBN) == B
        if self.niter == 0:
            best_rot = self.so3_base_rot[bestQ].to(self.device)
            best_trans = self.base_shifts[bestT].to(self.device)
        else:
            assert rot is not None
            best_rot = rot.view(-1, 8, 3, 3)[bestBN, bestQ]
            best_trans = trans.to(self.device)

        return best_rot, best_trans, new_init_poses
