import os.path as osp
from pathlib import Path
import warnings
from copy import deepcopy
import collections

import einops
import numpy as np
import biotite.structure as struc
import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary

import lightning.pytorch as pl
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy

from mmengine import mkdir_or_exist
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryodyna.utils.transforms import SpatialGridTranslate
# other
from cryodyna.utils.dataio import StarfileDataSet, StarfileDatasetConfig, Mask
from cryodyna.utils.ctf_utils import CTFRelion, CTFCryoDRGN
from cryodyna.utils.losses import calc_cor_loss, calc_kl_loss
from cryodyna.utils.misc import log_to_current, \
    pl_init_exp, pretty_dict, set_seed, warmup
from cryodyna.utils.pdb_tools import bt_save_pdb, extract_sec_ids_merge_small_blocks_CG, build_metagraph_knn_from_centroids, get_kplus_neighbor_metanodes
from cryodyna.gmm.gmm import EMAN2Grid, batch_projection, Gaussian
from cryodyna.gmm.deformer import E3Deformer, NMADeformer
from cryodyna.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryodyna.utils.polymer import Polymer, NT_ATOMS, AA_ATOMS
from cryodyna.utils.dist_loss_cg import (find_bonded_pairs, find_quaint_cutoff_pairs,
                                         calc_dist_by_pair_indices, find_nonbonded_pairs, 
                                         filter_same_chain_pairs, BondLoss, MorseLoss,
                                         remove_duplicate_pairs, remove_duplicate_pairs_Morse, DistLoss, find_angle_pairs, find_torsion_pairs, 
                                         find_improper_torsion_pairs, TorsionLoss, AngleLoss, ImproperTorsionLoss, find_quaint_cutoff_pairs_common)
from cryodyna.utils.latent_space_utils import get_nearest_point, cluster_kmeans, run_pca, get_pc_traj, run_umap
from cryodyna.utils.vis_utils import plot_z_dist, save_tensor_image
from cryodyna.utils.pl_utils import merge_step_outputs, squeeze_dict_outputs_1st_dim, \
    filter_outputs_by_indices, get_1st_unique_indices
from cryodyna.utils.align import _get_rmsd_loss
from scipy.spatial.distance import cdist
from miscs import calc_pair_dist_loss, calc_clash_loss, low_pass_mask2d, VAE, infer_ctf_params_from_config
from cryodyna.martini.IO import streamTag, pdbFrame, pdbChains, residues, Chain, pdbOut, pdbBoxString
from cryodyna.martini.TOP import Topology

# avoid num_workers set as cpu_count warning
warnings.simplefilter("ignore", PossibleUserWarning)

# only log to rank_zero, comment this for debugging
log_to_current = rank_zero_only(log_to_current)

TASK_NAME = "atom"


def prepare_images(images: torch.FloatTensor, space: str):
    assert space in ("real", "fourier")
    if space == "real":
        model_input = einops.rearrange(images, "b 1 ny nx -> b (1 ny nx)")
    else:
        fimages = primal_to_fourier_2d(images)
        model_input = einops.rearrange(torch.view_as_real(fimages), "b 1 ny nx c2 -> b (1 ny nx c2)", c2=2)
    return model_input


class InitTask(pl.LightningModule):

    def __init__(self, em_module):
        super().__init__()
        self.cfg = em_module.cfg
        self.em_module = em_module
        self.loss_deque = collections.deque([
            10,
        ], maxlen=20)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.loss_deque.append(outputs['loss'].item())
        if np.mean(self.loss_deque) <= 1e-3:
            self.trainer.should_stop = True
        # update all process status
        self.trainer.should_stop = self.trainer.strategy.broadcast(self.trainer.should_stop)

    def training_step(self, batch, batch_idx):
        images = batch["proj"]
        idxes = batch["idx"]
        rot_mats, trans_mats = self.em_module.get_batch_pose(batch)

        pred_deformation, mu = self.em_module.model(prepare_images(images, self.cfg.model.input_space))

        shift_loss = torch.mean(torch.pow(pred_deformation.flatten(start_dim=-2), 2))
        loss = shift_loss
        if self.global_step % self.cfg.runner.log_every_n_step == 0:
            log_to_current(f"loss {loss.item()}")
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.em_module.model.parameters(), lr=1e-4)

    def on_fit_end(self):
        log_to_current(f"Init finished with loss {np.mean(self.loss_deque)}")


class CryoEMTask(pl.LightningModule):

    def __init__(self, cfg, dataset):
        super().__init__()
        cfg = deepcopy(cfg)
        self.cfg = cfg

        # Define GMM
        inStream = streamTag(cfg.dataset_attr.ref_pdb_path)
        fileType = next(inStream)
        if fileType != "PDB":
            raise NotImplementedError
        title, atoms, box = pdbFrame(inStream)
        self.title = title
        self.box = box
        chains = [Chain([i for i in residues(chain)]) for chain in pdbChains(atoms)]
        n = 1
        log_to_current("Found %d chains:" % len(chains))
        for chain in chains:
            log_to_current("  %2d:   %s (%s), %d atoms in %d residues." % (n, chain.id, chain._type, chain.natoms, len(chain)))
            n += 1
        # Check all chains
        keep = []
        for chain in chains:
            if chain.type() == "Water":
                log_to_current("Removing %d water molecules (chain %s)." % (len(chain), chain.id))
            elif chain.type() in ("Protein", "Nucleic"):
                keep.append(chain)
            # This is currently not active:
            # elif options['RetainHETATM']:
            #     keep.append(chain)
            # else:
            #     log_to_current.info("Removing HETATM chain %s consisting of %d residues." % (chain.id, len(chain)))
        chains = keep
        # Get the total length of the sequence
        seqlength = sum([len(chain) for chain in chains])
        log_to_current('Total size of the system: %s residues.' % seqlength)

        coarseGrained, electrons = [], []
        coords, backbone_coords = [], []
        num_beads, chain_id, res_id, res_id_non_rb = [], [], [], []
        cum_len = 0

        for chain in chains:
            cg_atoms, elc = chain.cg(com=True)
            if not cg_atoms:
                raise ValueError(
                    f"No mapping for coarse graining chain {chain.id} ({chain.type()}); chain is skipped."
                )
            coarseGrained.extend(cg_atoms)
            electrons.extend(elc)   
            num_beads.append(len(cg_atoms))

            res_id_per_chain = []

            for name, resn, resi, ch, x, y, z in cg_atoms:
                resi_clean = resi & ((1 << 20) - 1)

                coords.append([x, y, z])
                if name in ("BB3", "BB"):
                    backbone_coords.append([x, y, z])

                chain_id.append(ch)
                res_id.append(resi_clean)

                res_id_per_chain.append(resi_clean)
                res_id_non_rb.append(cum_len + len(set(res_id_per_chain)))

            cum_len = cum_len + len(set(res_id_per_chain))

        coords = torch.tensor(coords)
        backbone_coords = torch.tensor(backbone_coords)

        if cfg.dataset_attr.ref_cg_pdb_path is not None:
            coords_ref = []
            backbone_coords_ref = []
            inStream = streamTag(cfg.dataset_attr.ref_cg_pdb_path)
            fileType = next(inStream)
            if fileType != "PDB":
                raise NotImplementedError
            _, atoms, _ = pdbFrame(inStream)
            for atom in atoms:
                coords_ref.append(atom[4:])
                if atom[0] == "BB3" or atom[0] == "BB":
                    backbone_coords_ref.append(atom[4:])
            coords_ref = torch.tensor(coords_ref)
            backbone_coords_ref = torch.tensor(backbone_coords_ref)
            assert coords_ref.shape == coords.shape
            assert backbone_coords_ref.shape == backbone_coords.shape
            align =  _get_rmsd_loss(coords.unsqueeze(0), coords_ref.unsqueeze(0))['align']
            coords = align(coords_ref)[0]
            backbone_coords = align(backbone_coords_ref)[0]
            
            log_to_current(f"Load reference structure from {cfg.dataset_attr.ref_cg_pdb_path}")
        else:
            log_to_current(f"Load reference structure from {cfg.dataset_attr.ref_pdb_path}")

        chain_id = np.array(chain_id)
        res_id = np.array(res_id)
        res_id_non_rb = np.array(res_id_non_rb)-1
        
        # for save
        self.template_pdb = coarseGrained

        backbone_coords = backbone_coords.float()        
        
        ss = []
        if cfg.cg_attr.dssp is not None:
            ss_method, ss_executable = "dssp", cfg.cg_attr.dssp
        else:
            log_to_current("No secondary structure or determination method speficied. Protein chains will be set to 'COIL'.")
            ss_method, ss_executable = None, None
        for chain in chains:
            s = chain.dss(ss_method, ss_executable)
            ss += s
        ss = "".join(ss)
        ss_per_chain = []
            
        tops = []
        sizes = []
        for i, mol in enumerate(chains):
            name = mol.getname()
            top = Topology(mol, name=name)
            tops.append(top)
            ss_per_chain.append(top.secstruc)
            sizes.extend(top.get_bead_size())
         # ref + cg
        ref_centers = coords.float()
        ref_amps = torch.tensor(electrons).float()
        ref_sigmas = torch.tensor(sizes).float()*10/2
        num_pts = ref_centers.shape[0]
        log_to_current(f"Reference structure has {num_pts} bead coordinates")
        log_to_current(f"1st GMM blob amplitude {ref_amps[0].item()}, mean sigma {ref_sigmas.mean().item()}")

        sec_ids = extract_sec_ids_merge_small_blocks_CG(ss_per_chain,min_block_len=3)
        meta_edge_index, centroids = build_metagraph_knn_from_centroids(pos=backbone_coords,sec_ids=sec_ids,k=cfg.knn_num)

        centroid_distances = cdist(centroids, centroids)
        edge_dist = centroid_distances[meta_edge_index[0], meta_edge_index[1]]
        pe_vector_res_2_meta = [centroids[sec_id] - backbone_coords[i] for i,sec_id in enumerate(sec_ids)]
        pe_vector_bead_2_res = [backbone_coords[residue_id] - ref_centers[i] for i,residue_id in enumerate(res_id_non_rb)]
        meta_2_node_edge, meta_2_node_vector = get_kplus_neighbor_metanodes(backbone_coords,sec_ids,centroids)
        
        struc_feature = {
            'sec_ids': sec_ids,
            'res_id_non_rb': res_id_non_rb,
            'meta_edge_index': meta_edge_index,
            'edge_dist': edge_dist,
            'pe_vector_res_2_meta': pe_vector_res_2_meta,
            'pe_vector_bead_2_res': pe_vector_bead_2_res,
            'meta_2_node_edge': meta_2_node_edge,
            'meta_2_node_vector': meta_2_node_vector
        }

        pickle.dump(struc_feature,open(f'{self.cfg.work_dir}/struc_fea.pkl','wb'))
        # tunable params
        # gmm
        self.register_buffer("gmm_centers", ref_centers)

        if cfg.gmm.tunable:
            log_to_current("Set GMM sigmas, amplitudes tunable")
            self.register_parameter("gmm_sigmas", nn.Parameter(ref_sigmas))
            self.register_parameter("gmm_amps", nn.Parameter(ref_amps))
        else:
            self.register_buffer("gmm_sigmas", ref_sigmas)
            self.register_buffer("gmm_amps", ref_amps)

        nma_modes = None
        if (hasattr(self.cfg.extra_input_data_attr, "nma_path") and
                self.cfg.extra_input_data_attr.nma_path not in ["", None]):
            nma_modes = torch.tensor(np.load(self.cfg.extra_input_data_attr.nma_path), dtype=torch.float32)
            log_to_current(f"Load NMA coefficients from {self.cfg.extra_input_data_attr.nma_path}, "
                           f"whose shape is {nma_modes.shape}")

        # model
        if cfg.model.input_space == "fourier":
            in_dim = 2 * cfg.data_process.down_side_shape ** 2
        elif cfg.model.input_space == "real":
            in_dim = cfg.data_process.down_side_shape ** 2
        else:
            raise NotImplementedError
        self.model = VAE(in_dim=in_dim,
                        out_dim=num_pts * 3 if nma_modes is None else 6 + nma_modes.shape[1],
                        sec_ids=torch.from_numpy(np.array(sec_ids)).long(),
                        meta_edge_index=meta_edge_index.long(),
                        edge_dist=torch.from_numpy(edge_dist).float(),
                        meta_2_node_edge=meta_2_node_edge.long(),
                        meta_2_node_vector=meta_2_node_vector,
                        beads_ids=torch.from_numpy(res_id_non_rb),
                        pe_vector_res_2_meta=torch.stack(pe_vector_res_2_meta,dim=0),
                        pe_vector_bead_2_res=torch.stack(pe_vector_bead_2_res,dim=0),
                        **cfg.model.model_cfg)

        if nma_modes is None:
            self.deformer = E3Deformer()
        else:
            self.deformer = NMADeformer(nma_modes)

        # loss or regularization's preparation
        # dist loss
        connect_pairs, bond_ref_lengths, bond_kbs = find_bonded_pairs(tops, num_beads)


        if "Nucleic" in [chain.type() for chain in chains]:
            chain_id_unique = np.array(list(dict.fromkeys(chain_id)))
            aa_chain_mask = np.array([chain.type() for chain in chains]) == 'Protein'
            tmp_mask= np.isin(chain_id, chain_id_unique[aa_chain_mask])
            aa_indices_in_pdb = np.nonzero(tmp_mask)[0]
            aa_coords = coords[tmp_mask]
            aa_chain_id = chain_id[tmp_mask]
            aa_res_id = res_id[tmp_mask]
            aa_cutoff_pairs = find_quaint_cutoff_pairs(aa_coords, aa_chain_id, aa_res_id,
                                                        cfg.loss.intra_chain_cutoff,
                                                        cfg.loss.inter_chain_cutoff, cfg.loss.intra_chain_res_bound)
            aa_cutoff_pairs = aa_indices_in_pdb[aa_cutoff_pairs]
            log_to_current(f"{len(aa_cutoff_pairs)} AA pairs")

            nt_chain_mask = np.array([chain.type() for chain in chains]) == 'Nucleic'
            tmp_mask= np.isin(chain_id, chain_id_unique[nt_chain_mask])
            nt_indices_in_pdb = np.nonzero(tmp_mask)[0]
            nt_coords = coords[tmp_mask]
            nt_chain_id = chain_id[tmp_mask]
            nt_res_id = res_id[tmp_mask]
            nt_cutoff_pairs = find_quaint_cutoff_pairs(nt_coords, nt_chain_id, nt_res_id,
                                                        cfg.loss.nt_intra_chain_cutoff,
                                                        cfg.loss.nt_inter_chain_cutoff, cfg.loss.nt_intra_chain_res_bound)
            nt_cutoff_pairs = nt_indices_in_pdb[nt_cutoff_pairs]
            log_to_current(f"{len(nt_cutoff_pairs)} NT pairs")

            aa_nt_cutoff_pairs = find_quaint_cutoff_pairs_common(aa_coords, nt_coords,
                                                                cfg.loss.inter_chain_cutoff_aa_nt,)
            aa_nt_cutoff_pairs = np.column_stack((aa_indices_in_pdb[aa_nt_cutoff_pairs[:, 0]], nt_indices_in_pdb[aa_nt_cutoff_pairs[:, 1]])) # N, 2
            log_to_current(f"{len(aa_nt_cutoff_pairs)} AA-NT pairs")

            cutoff_pairs = np.vstack((aa_cutoff_pairs, nt_cutoff_pairs))
        else:
            cutoff_pairs = find_quaint_cutoff_pairs(coords, chain_id, res_id,
                                                    cfg.loss.intra_chain_cutoff, cfg.loss.inter_chain_cutoff,
                                                    cfg.loss.intra_chain_res_bound)
            aa_nt_cutoff_pairs = []

        nonbonded_pairs, epsilons, sigmas = find_nonbonded_pairs(tops, num_beads, coords, chain_id, cfg.loss.nonbonded_cutoff)
        angle_pairs, angle_ref_lengths, angle_kbs = find_angle_pairs(tops, num_beads)
        torsion_pairs, torsion_ref_lengths, torsion_kbs, torsion_muls = find_torsion_pairs(tops, num_beads)
        improper_pairs, improper_ref_lengths, improper_kbs = find_improper_torsion_pairs(tops, num_beads)
        cutoff_pairs = remove_duplicate_pairs(cutoff_pairs, connect_pairs)
        nonbonded_pairs, epsilons, sigmas = remove_duplicate_pairs_Morse(nonbonded_pairs, epsilons, sigmas, connect_pairs)

        if len(cutoff_pairs) > 0:
            log_to_current(f"found {len(cutoff_pairs)} cutoff_pairs")

            dists = calc_dist_by_pair_indices(coords, cutoff_pairs)
            self.dist_loss_fn = DistLoss(cutoff_pairs, dists, reduction=None)
            # for chain-wise dropout
            cutoff_chain_mask = filter_same_chain_pairs(cutoff_pairs, chain_id)
            self.register_buffer("cutoff_chain_mask", torch.from_numpy(cutoff_chain_mask))
        else:
            log_to_current("cutoff_pairs is empty")
        
        if len(aa_nt_cutoff_pairs) > 0:
            log_to_current(f"found {len(aa_nt_cutoff_pairs)} AA-NT pairs")
            dists = calc_dist_by_pair_indices(coords, aa_nt_cutoff_pairs)
            self.aa_nt_dist_loss_fn = DistLoss(aa_nt_cutoff_pairs, dists, reduction=None)
        else:
            log_to_current("aa_nt_cutoff_pairs is empty")

        if len(connect_pairs) > 0:
            self.register_buffer("connect_pairs", torch.from_numpy(connect_pairs).long())
            self.register_buffer("bond_ref_lengths", torch.from_numpy(bond_ref_lengths).float())
            self.register_buffer("bond_kbs", torch.from_numpy(bond_kbs).float())
            self.bond_loss_fn = BondLoss(self.connect_pairs, self.bond_ref_lengths, len(connect_pairs), self.bond_kbs)
            log_to_current(f"found {len(connect_pairs)} connect_pairs")
        else:
            log_to_current("connect_pairs is empty")

        if len(angle_pairs) > 0:
            self.register_buffer("angle_pairs", torch.from_numpy(angle_pairs).long())
            self.register_buffer("angle_ref_lengths", torch.from_numpy(angle_ref_lengths).float())
            self.register_buffer("angle_kbs", torch.from_numpy(angle_kbs).float())
            self.angle_loss_fn = AngleLoss(self.angle_pairs, self.angle_ref_lengths, num_pts**2, self.angle_kbs)
            log_to_current(f"found {len(angle_pairs)} angle_pairs")
        else:
            log_to_current("angle_pairs is empty")

        if len(torsion_pairs) > 0:
            self.register_buffer("torsion_pairs", torch.from_numpy(torsion_pairs).long())
            self.register_buffer("torsion_ref_lengths", torch.from_numpy(torsion_ref_lengths).float())
            self.register_buffer("torsion_kbs", torch.from_numpy(torsion_kbs).float())
            self.register_buffer("torsion_muls", torch.from_numpy(torsion_muls).float())
            self.torsion_loss_fn = TorsionLoss(self.torsion_pairs, self.torsion_ref_lengths, self.torsion_muls, num_pts**2, self.torsion_kbs)
            log_to_current(f"found {len(torsion_pairs)} torsion_pairs")
        else:
            log_to_current("torsion_pairs is empty")
        
        if len(improper_pairs) > 0:
            self.register_buffer("improper_pairs", torch.from_numpy(improper_pairs).long())
            self.register_buffer("improper_ref_lengths", torch.from_numpy(improper_ref_lengths).float())
            self.register_buffer("improper_kbs", torch.from_numpy(improper_kbs).float())
            self.improper_loss_fn = ImproperTorsionLoss(self.improper_pairs, self.improper_ref_lengths, num_pts**2, self.improper_kbs)
 
            log_to_current(f"found {len(improper_pairs)} improper_pairs")
        else:
            log_to_current("improper_pairs is empty")
        
        if len(nonbonded_pairs) > 0:
            self.register_buffer("epsilons", torch.from_numpy(epsilons).float())
            self.register_buffer("sigmas", torch.from_numpy(sigmas).float())
            self.register_buffer("nonbonded_pairs", torch.from_numpy(nonbonded_pairs).long())

            self.nonbond_loss_fn = MorseLoss(self.nonbonded_pairs, self.epsilons, self.sigmas, len(nonbonded_pairs), a=cfg.loss.Morse_a)

            log_to_current(f"found {len(nonbonded_pairs)} nonbonded_pairs")
        else:
            log_to_current("nonbonded_pairs is empty")
            

        # low-pass filtering
        if hasattr(cfg.data_process, "low_pass_bandwidth"):
            log_to_current(f"Use low-pass filtering w/ {cfg.data_process.low_pass_bandwidth} A")
            lp_mask2d = low_pass_mask2d(cfg.data_process.down_side_shape, cfg.data_process.down_apix,
                                        cfg.data_process.low_pass_bandwidth)
            self.register_buffer("lp_mask2d", torch.from_numpy(lp_mask2d).float())
        else:
            self.lp_mask2d = None


        self.mask = Mask(cfg.data_process.down_side_shape, rad=cfg.loss.mask_rad_for_image_loss)

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
        # cache
        self.validation_step_outputs = []
        self.stored_metrics = {}
        self.history_saved_dirs = []
        self.training_step_outputs = []
        if getattr(self.cfg.extra_input_data_attr, "ckpt_path", None) is not None:
            log_to_current(f"load checkpoint from {self.cfg.extra_input_data_attr.ckpt_path}")
            self._load_ckpt(self.cfg.extra_input_data_attr.ckpt_path)

    def _save_ckpt(self, ckpt_path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "gmm_sigmas": self.gmm_sigmas.data,
                "gmm_amps": self.gmm_amps.data
            }, ckpt_path)

    def _load_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(state_dict["model"])
        if self.cfg.gmm.tunable:
            self.gmm_sigmas.data = state_dict["gmm_sigmas"]
            self.gmm_amps.data = state_dict["gmm_amps"]

    def _get_save_dir(self):
        save_dir = osp.join(self.cfg.work_dir, f"{self.current_epoch:04d}_{self.global_step:07d}")
        mkdir_or_exist(save_dir)
        return save_dir

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

    def _shared_forward(self, images):
        # predict structure
        pred_deformation, mu = self.model(prepare_images(images, self.cfg.model.input_space))
        return pred_deformation, mu

    def _shared_projection(self, pred_struc, rot_mats):
        pred_images = batch_projection(
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
        pred_ctf_params = {k: batch[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism') if k in batch}
        f_proj = self.ctf(f_proj, batch['idx'], ctf_params=pred_ctf_params, mode="gt", frequency_marcher=None)
        if freq_mask is not None:
            f_proj = f_proj * self.lp_mask2d
        return f_proj

    def _shared_infer(self, batch):
        gt_images = batch["proj"]
        idxes = batch["idx"]

        rot_mats, trans_mats = self.get_batch_pose(batch)

        # if self.lp_mask2d is not None:
        #     gt_images = self.low_pass_images(gt_images)

        # prediction
        pred_deformation, mu = self._shared_forward(gt_images)
        pred_struc = self.deformer.transform(pred_deformation, self.gmm_centers)

        # get gmm projections
        pred_gmm_images = self._shared_projection(pred_struc, rot_mats)

        # apply ctf, low-pass
        pred_gmm_images = self._apply_ctf(batch, pred_gmm_images, self.lp_mask2d)

        if trans_mats is not None:
            gt_images = self.translator.transform(einops.rearrange(gt_images, "B 1 NY NX -> B NY NX"),
                                                  einops.rearrange(trans_mats, "B C2 -> B 1 C2"))

        return gt_images, pred_gmm_images, pred_struc, mu

    def _shared_decoding(self, z):
        with torch.no_grad():
            z = z.float().to(self.device)

            pred_deformation = self.model.decoder(z)
            pred_struc = self.deformer.transform(pred_deformation, self.gmm_centers)
            pred_struc = pred_struc.squeeze(0)
        return pred_struc

    def _save_batched_strucs(self, pred_strucs, save_path):
        ref_cg = self.template_pdb.copy()
        b = pred_strucs.shape[0]
        cgOutPDB = open(save_path, "w")
        
        for i in range(b):
            atid = 1
            cgOutPDB.write(f"MODEL {i+1}\n")
            cgOutPDB.write(self.title)
            cgOutPDB.write(pdbBoxString(self.box))
            tmp_struc = pred_strucs[i].cpu().numpy()
            for j, (name, resn, resi, chain, x_ref, y_ref, z_ref) in enumerate(ref_cg):
                x, y, z = tmp_struc[j]
                cgOutPDB.write(pdbOut((name, resn[:3], resi, chain, x, y, z),i=atid))
                atid += 1
            cgOutPDB.write("TER\n")
            cgOutPDB.write("ENDMDL\n")

        cgOutPDB.close()
        
    def _shared_image_check(self, total=25):
        mode = self.model.training
        # use validation or test set which not shuffled
        tmp_loader = self.trainer.val_dataloaders or self.trainer.test_dataloaders

        num = 0
        gt_images_list = []
        pred_gmm_images_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in tmp_loader:
                batch = self.trainer.strategy.batch_to_device(batch)

                gt_images, pred_gmm_images, _, mu= self._shared_infer(batch)

                gt_images_list.append(gt_images)
                pred_gmm_images_list.append(pred_gmm_images)

                num += gt_images.shape[0]
                if num >= total:
                    break

        self.model.train(mode=mode)

        gt_images_list = torch.cat(gt_images_list, dim=0)[:total]
        pred_gmm_images_list = torch.cat(pred_gmm_images_list, dim=0)[:total]

        save_dir = self._get_save_dir()

        save_tensor_image(gt_images_list, osp.join(save_dir, "input_image.png"))
        save_tensor_image(pred_gmm_images_list, osp.join(save_dir, "pred_gmm_image.png"), self.mask.mask)

    # standard hooks:
    def training_step(self, batch, batch_idx):
        cfg = self.cfg

        gt_images, pred_gmm_images, pred_struc, mu= self._shared_infer(batch)
        if torch.isnan(pred_struc).any():
            print("batch_struc contains NaN!")
            nan_indices = torch.nonzero(torch.isnan(pred_struc))
            print("NaN indices:", nan_indices)
        # gmm part loss
        # only gmm supervision should be low-passed
        if self.lp_mask2d is not None:
            lp_gt_images = self.low_pass_images(gt_images)
        else:
            lp_gt_images = gt_images
        gmm_proj_loss = calc_cor_loss(pred_gmm_images, lp_gt_images, self.mask)
        weighted_gmm_proj_loss = cfg.loss.gmm_cryoem_weight * gmm_proj_loss

        if hasattr(self, "connect_pairs"):
            # max_epochs = self.trainer.max_epochs
            # current_epoch = self.current_epoch
            # beta = 1 * (current_epoch / max_epochs)
            # bonded_loss = self.bond_loss_fn(pred_struc, beta)
            bonded_loss = self.bond_loss_fn(pred_struc)
            weighted_bonded_loss = cfg.loss.bonded_weight * bonded_loss
        else:
            weighted_bonded_loss = weighted_gmm_proj_loss.new_tensor(0.)
        if hasattr(self, "angle_pairs"):
            angle_loss = self.angle_loss_fn(pred_struc)
            weighted_angle_loss = cfg.loss.angle_weight * angle_loss
        else:
            weighted_angle_loss = weighted_gmm_proj_loss.new_tensor(0.)
        
        if hasattr(self, "torsion_pairs"):
            torsion_loss = self.torsion_loss_fn(pred_struc)
            weighted_torsion_loss = cfg.loss.torsion_weight * torsion_loss
        else:
            weighted_torsion_loss = weighted_gmm_proj_loss.new_tensor(0.)
        
        if hasattr(self, "improper_pairs"):
            improper_loss = self.improper_loss_fn(pred_struc)
            weighted_improper_loss = cfg.loss.improper_weight * improper_loss
        else:
            weighted_improper_loss = weighted_gmm_proj_loss.new_tensor(0.)

        if hasattr(self, "nonbonded_pairs"):
            nonbonded_loss = self.nonbond_loss_fn(pred_struc)
            weighted_nonbonded_loss = cfg.loss.nonbonded_weight * nonbonded_loss
        else:
            weighted_nonbonded_loss = weighted_gmm_proj_loss.new_tensor(0.)
            
        if hasattr(self, "dist_loss_fn"):
            dist_loss = self.dist_loss_fn(pred_struc)
            # across devices
            all_dist_loss = self.all_gather(dist_loss)  # world_size, batch, num_pairs
            all_dist_loss = all_dist_loss.reshape(-1, dist_loss.shape[-1])

            # chain-wise drop
            with torch.no_grad():
                keep_mask = torch.ones(dist_loss.shape[-1], dtype=torch.bool).to(dist_loss.device)

                for i in range(len(self.cutoff_chain_mask)):
                    tmp_mask = self.cutoff_chain_mask[i]
                    tmp_var = all_dist_loss.index_select(dim=1, index=tmp_mask.nonzero(as_tuple=True)[0]).var(dim=0)
                    intra_chain_keep_mask = tmp_var.lt(torch.quantile(tmp_var, cfg.loss.dist_keep_ratio))
                    keep_mask[tmp_mask] *= intra_chain_keep_mask
                keep_mask = keep_mask.unsqueeze(0).repeat(dist_loss.size(0), 1)
            
            dist_loss = torch.mean(dist_loss[keep_mask])

            weighted_dist_loss = cfg.loss.dist_weight * dist_loss
        else:
            weighted_dist_loss = weighted_gmm_proj_loss.new_tensor(0.)
        
        if hasattr(self, "aa_nt_dist_loss_fn"):
            aa_nt_dist_loss = self.aa_nt_dist_loss_fn(pred_struc)
            all_aa_nt_dist_loss = self.all_gather(aa_nt_dist_loss)
            all_aa_nt_dist_loss = all_aa_nt_dist_loss.reshape(-1, aa_nt_dist_loss.shape[-1])
            with torch.no_grad():
                tmp_var = all_aa_nt_dist_loss.var(dim=0)
                keep_mask = tmp_var.lt(torch.quantile(tmp_var, cfg.loss.dist_keep_ratio_aa_nt))
            aa_nt_dist_loss = torch.mean(aa_nt_dist_loss[keep_mask.unsqueeze(0).repeat(aa_nt_dist_loss.size(0), 1)])
            weighted_aa_nt_dist_loss = cfg.loss.dist_aa_nt_weight * aa_nt_dist_loss
        else:
            weighted_aa_nt_dist_loss = weighted_gmm_proj_loss.new_tensor(0.)

        loss = (weighted_gmm_proj_loss + weighted_bonded_loss + weighted_angle_loss + weighted_torsion_loss + weighted_improper_loss + weighted_nonbonded_loss + weighted_dist_loss + weighted_aa_nt_dist_loss)

        tmp_metric = {
            "loss": loss.item(),
            "cryoem(gmm)": weighted_gmm_proj_loss.item(),
            "bonded": weighted_bonded_loss.item(),
            "angle": weighted_angle_loss.item(),
            "torsion": weighted_torsion_loss.item(),
            "improper": weighted_improper_loss.item(),
            "nonbonded": weighted_nonbonded_loss.item(),
            "dist": weighted_dist_loss.item(),
            "dist_aa_nt": weighted_aa_nt_dist_loss.item(),
        }

        self.training_step_outputs.append({"cryoem(gmm)": tmp_metric['cryoem(gmm)'], "bonded":tmp_metric['bonded'], "angle":tmp_metric['angle'], "torsion":tmp_metric['torsion'], "improper":tmp_metric['improper'], "nonbonded":tmp_metric['nonbonded'], "dist":tmp_metric['dist'], "dist_aa_nt":tmp_metric['dist_aa_nt']})
        
        if self.global_step % cfg.runner.log_every_n_step == 0:
            self.log_dict(tmp_metric)
            log_to_current(f"epoch {self.current_epoch} [{batch_idx}/{self.trainer.num_training_batches}] | " +
                           pretty_dict(tmp_metric, 5))
        return loss

    def validation_step(self, batch, batch_idx):
        gt_images = batch["proj"]
        idxes = batch["idx"]

        # if self.lp_mask2d is not None:
        #     gt_images = self.low_pass_images(gt_images)

        mu = self.model.encode(prepare_images(gt_images, self.cfg.model.input_space))
        z = mu
        self.validation_step_outputs.append({"z": z, "idx": idxes})
    
    def on_train_epoch_end(self):
        step_num = len(self.training_step_outputs)
        loss_record = {ls:0 for ls in self.training_step_outputs[0].keys()}
        for step in self.training_step_outputs:
            for loss in loss_record.keys():
                loss_record[loss] += step[loss]
        loss_record = {key: float(metric / step_num) for key,metric in loss_record.items()}
        log_to_current(f"epoch {self.current_epoch} Average | " +
                           pretty_dict(loss_record, 6))
        self.training_step_outputs.clear() 
    
    def on_validation_epoch_end(self):
        # lightning will automatically copy val samples to let val_loader to be divided by gpu_num with no remainder,
        # here use sample id to remove redundancy
        all_outputs = merge_step_outputs(self.validation_step_outputs)
        all_outputs = self.all_gather(all_outputs)
        all_outputs = squeeze_dict_outputs_1st_dim(all_outputs)

        if self.trainer.is_global_zero and len(all_outputs) > 0:
            # save projection images for checking
            self._shared_image_check()

            save_dir = self._get_save_dir()

            # --------
            # dealing with all z
            indices = get_1st_unique_indices(all_outputs["idx"])
            log_to_current(f"Total {len(indices)} unique samples")
            all_outputs = filter_outputs_by_indices(all_outputs, indices)
            z_list = all_outputs["z"]
            z_list = z_list.cpu().numpy()  # (num_samples, latent_dim)

            np.save(osp.join(save_dir, "z.npy"), z_list)

            # --------
            # Kmeans cluster
            kmeans_labels, centers = cluster_kmeans(z_list, self.cfg.analyze.cluster_k)
            centers, centers_ind = get_nearest_point(z_list, centers)

            if z_list.shape[-1] > 2 and not self.cfg.analyze.skip_umap:
                log_to_current("Running UMAP...")
                z_emb, reducer = run_umap(z_list)
                centers_emb = reducer.transform(centers)
                try:
                    plot_z_dist(z_emb, extra_cluster=centers_emb, save_path=osp.join(save_dir, "z_distribution.png"))
                except Exception as e:
                    log_to_current(e)

            elif z_list.shape[-1] <= 2:
                try:
                    plot_z_dist(z_list, extra_cluster=centers, save_path=osp.join(save_dir, "z_distribution.png"))
                except Exception as e:
                    log_to_current(e)
            else:
                #
                pass

            centers = torch.from_numpy(centers)
            pred_struc = self._shared_decoding(centers)

            self._save_batched_strucs(pred_struc, f"{save_dir}/pred.pdb")

            # --------
            # pca
            for pca_dim in range(1, 1 + min(3, self.cfg.model.model_cfg.latent_dim)):
                pc, pca = run_pca(z_list)
                start = np.percentile(pc[:, pca_dim - 1], 5)
                stop = np.percentile(pc[:, pca_dim - 1], 95)
                z_pc_traj = get_pc_traj(pca, z_list.shape[1], 10, pca_dim, start, stop)
                z_pc_traj, _ = get_nearest_point(z_list, z_pc_traj)

                z_pc_traj = torch.from_numpy(z_pc_traj)
                pred_struc = self._shared_decoding(z_pc_traj)

                self._save_batched_strucs(pred_struc, f"{save_dir}/pca-{pca_dim}.pdb")

        # important
        self.trainer.strategy.barrier()
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        params = [*self.model.parameters()]
        if self.cfg.gmm.tunable:
            params.extend([self.gmm_sigmas, self.gmm_amps])
        if hasattr(self, "dist_loss_fn"):
            params.extend(self.dist_loss_fn.parameters())
        optimizer = optim.AdamW(params, lr=self.cfg.optimizer.lr)
        # torch.nn.utils.clip_grad_norm_(params, 1.0)
        return optimizer

    # extra hooks:
    # here self.device is set to cuda:0, 1 etc
    def on_fit_start(self):
        self.cfg.work_dir = self.trainer.strategy.broadcast(self.cfg.work_dir)

        # make sure model parameters are the same
        log_to_current(f"load rank 0 model weights")
        state_dict = self.trainer.strategy.broadcast(self.model.state_dict())
        self.model.load_state_dict(state_dict)

    def on_validation_start(self):
        log_to_current(f"Epoch {self.current_epoch} Step {self.global_step} start validation")
        state_dict = self.trainer.strategy.broadcast(self.model.state_dict())
        self.model.load_state_dict(state_dict)

        if self.trainer.is_global_zero:
            save_dir = self._get_save_dir()
            self._save_ckpt(osp.join(save_dir, "ckpt.pt"))
            self.history_saved_dirs.append(save_dir)
            # keep_last_k = 1
            # if len(self.history_saved_dirs) >= keep_last_k:
            #     for to_remove in self.history_saved_dirs[:-keep_last_k]:
            #         p = Path(to_remove) / "ckpt.pt"
            #         if p.exists():
            #             p.unlink()
            #             log_to_current(f"delete {p} to keep last {keep_last_k} ckpts")

    def on_train_start(self):
        if self.trainer.is_global_zero:
            self._shared_image_check()
        self.trainer.strategy.barrier()


def train():
    cfg = pl_init_exp(exp_prefix=TASK_NAME, backup_list=[
        __file__,
    ], inplace=False)

    if cfg.seed is not None:
        set_seed(cfg.seed)
        log_to_current(f"seed set to {cfg.seed}")

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

    #
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

    rank_zero_only(cfg.dump)(osp.join(cfg.work_dir, "config.py"))

    log_to_current(f"Load dataset from {dataset.cfg.dataset_dir}, power scaled by {dataset.cfg.power_images}")
    log_to_current(f"Total {len(dataset)} samples")
    log_to_current(f"The dataset side_shape: {dataset.side_shape}, apix: {dataset.apix}")
    log_to_current(f"Set down-sample side_shape {dataset.down_side_shape} with apix {cfg.data_process.down_apix}")

    train_loader = DataLoader(dataset,
                              batch_size=cfg.data_loader.train_batch_per_gpu,
                              shuffle=True,
                              drop_last=True,
                              num_workers=cfg.data_loader.workers_per_gpu)

    test_loader = DataLoader(dataset,
                             batch_size=cfg.data_loader.val_batch_per_gpu,
                             shuffle=False,
                             drop_last=False,
                             num_workers=cfg.data_loader.workers_per_gpu)

    em_task = CryoEMTask(cfg, dataset)
    if not cfg.eval_mode and cfg.do_ref_init:
        init_task = InitTask(em_task)
        # if you meet libibverbs warnings, try process_group_backend="gloo"
        init_trainer = pl.Trainer(max_epochs=3,
                                  devices=cfg.trainer.devices,
                                  accelerator="gpu" if torch.cuda.is_available() else "cpu",
                                  precision=cfg.trainer.precision,
                                  strategy=DDPStrategy(process_group_backend="nccl", find_unused_parameters=True),
                                  logger=False,
                                  enable_checkpointing=False,
                                  enable_model_summary=False,
                                  enable_progress_bar=False,
                                  num_sanity_val_steps=0)

        init_trainer.fit(init_task, train_dataloaders=train_loader)

    em_trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            # detect_anomaly=True,
                            # gradient_clip_val=0.5, 
                            # gradient_clip_algorithm="norm",
                            strategy=DDPStrategy(process_group_backend="nccl"),
                            logger=False,
                            enable_checkpointing=False,
                            enable_model_summary=False,
                            enable_progress_bar=False,
                            **cfg.trainer)

    if not cfg.eval_mode:
        em_trainer.fit(model=em_task, train_dataloaders=train_loader, val_dataloaders=test_loader)
    else:
        em_trainer.validate(model=em_task, dataloaders=test_loader)


if __name__ == "__main__":
    train()