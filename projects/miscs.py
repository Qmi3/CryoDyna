# Inherited structural regularization from CryoSTAR - https://github.com/bytedance/cryostar
# Copyright 2023 Bytedance Inc. - Apache License 2.0
from functools import lru_cache
from pathlib import Path

import einops
import numpy as np

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = np

import torch
from torch import linalg as LA
from torch import nn
import torch.nn.functional as F

from cryodyna.common.residue_constants import ca_ca
from cryodyna.utils.misc import log_to_current
from cryodyna.utils.ml_modules import VAEEncoder, EGNNDecoder, reparameterize,Decoder,  GATEncoder,HierarchicalDeltaGNN, MultiScaleGATEncoder, HierarchicalDeltaGNN_CG
from cryodyna.utils.ctf import parse_ctf_star
from lightning.pytorch.utilities import rank_zero_only
from typing import Union


CA_CA = round(ca_ca, 2)
log_to_current = rank_zero_only(log_to_current)


def infer_ctf_params_from_config(cfg):
    star_file_path = Path(cfg.dataset_attr.starfile_path)
    ctf_params = parse_ctf_star(star_file_path, side_shape=cfg.data_process.down_side_shape,
                                apix=cfg.data_process.down_apix)[0].tolist()
    ctf_params = {
        "size": cfg.data_process.down_side_shape,
        "resolution": cfg.data_process.down_apix,
        "kV": ctf_params[5],
        "cs": ctf_params[6],
        "amplitudeContrast": ctf_params[7]
    }
    return ctf_params


def low_pass_mask3d(shape, apix=1., bandwidth=2):
    freq = np.fft.fftshift(np.fft.fftfreq(shape, apix))
    freq = freq**2
    freq = np.sqrt(freq[:, None, None] + freq[None, :, None] + freq[None, None])

    mask = np.asarray(freq < 1 / bandwidth, dtype=np.float32)
    # trick to avoid "ringing", however you should increase sigma to about 11 to completely remove artifact
    # gaussian_filter(mask, 3, output=mask)
    return mask


def low_pass_mask2d(shape, apix=1., bandwidth=2):
    freq = np.fft.fftshift(np.fft.fftfreq(shape, apix))
    # why **2 maybe 2d要这么算 mask更高频信息
    freq = freq**2
    freq = np.sqrt(freq[:, None] + freq[None, :])
    mask = np.asarray((freq < 1 / bandwidth) , dtype=np.float32) 
    return mask


def get_edges_batch(edges,edge_attr,n_nodes, batch_size):
    
    edges = torch.LongTensor(edges)
    rows, cols = [], []
    for i in range(batch_size):
        rows.append(edges[:,0] + n_nodes * i)
        cols.append(edges[:,1] + n_nodes * i)
    edges = torch.stack([torch.cat(rows), torch.cat(cols)], dim=-1)
    edge_attr = torch.tensor(edge_attr).repeat(batch_size).unsqueeze(1)
    # return edges,edge_attr,edge_bond.unsqueeze(1)
    return edges,edge_attr

def calc_clash_loss(pred_struc, pair_index, clash_cutoff=4.0):
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    possible_clash_dist = pred_dist[pred_dist < clash_cutoff]
    if possible_clash_dist.numel() == 0:
        avg_loss = torch.tensor(0.0).to(pred_struc)
    else:
        possible_clash_loss = (clash_cutoff - possible_clash_dist)**2
        avg_loss = possible_clash_loss.mean()
    return avg_loss



def calc_pair_dist_loss(pred_struc, pair_index, target_dist, type="vanilla", chain_id=None):
    bsz = pred_struc.shape[0]
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    return F.mse_loss(pred_dist, target_dist.repeat(bsz, 1))


class VAE(nn.Module):

    def __init__(
        self,
        encoder_cls: str,
        decoder_cls: str,
        in_dim: int,
        out_dim: int,
        sec_ids: list,
        meta_edge_index: np.array,
        edge_dist:np.array,
        pe_vector:np.array,
        meta_2_node_edge:torch.tensor,
        meta_2_node_vector:torch.tensor,
        attention_layer:int,
        e_hidden_dim: Union[int, list, tuple],
        z_dim:int,
        latent_dim: int,
        d_hidden_dim:Union[int, list, tuple],
        e_hidden_layers: int,
        d_hidden_layers: int,
    ):
        super().__init__()
        if encoder_cls == "MLP":
            self.encoder = VAEEncoder(in_dim, e_hidden_dim, z_dim, e_hidden_layers)
        elif encoder_cls == "GAT":
            self.encoder = GATEncoder(in_dim, e_hidden_dim, attention_layer, z_dim, e_hidden_layers)
        elif encoder_cls == "MS-GAT":
            self.encoder = MultiScaleGATEncoder(in_dim, e_hidden_dim, attention_layer, z_dim, e_hidden_layers)
        else:
            raise Exception()
        if decoder_cls == "MLP":
            self.decoder = Decoder(z_dim, d_hidden_dim, out_dim, d_hidden_layers)
        elif decoder_cls == "GNN_with_prior":
            self.decoder = EGNNDecoder(d_hidden_dim, d_hidden_layers)
        elif decoder_cls == "metaGNN":
            self.decoder = HierarchicalDeltaGNN(in_dim = z_dim, d_hidden_dim = d_hidden_dim, latent_dim = latent_dim, sec_ids=sec_ids, meta_edge_index=meta_edge_index, 
                                                edge_dist=edge_dist, pe_vector = pe_vector,meta_2_node_edge = meta_2_node_edge, meta_2_node_vector = meta_2_node_vector,out_dim=out_dim)
        else:
            print(f"{decoder_cls} not in presets, you may set it manually later.")
            self.decoder: torch.nn.Module

    def encode(self,x):
        mean = self.encoder(x)
        return mean

    def forward(self, img):
        z = self.encode(img)
        out = self.decoder(z)
        return out, z
    
    def eval_z(self, z):
        out = self.decoder(z)
        return out

class VAE_CG(nn.Module):

    def __init__(
        self,
        encoder_cls: str,
        decoder_cls: str,
        in_dim: int,
        out_dim: int,
        sec_ids: list,
        beads_ids: list,
        meta_edge_index: np.array,
        edge_dist:np.array,
        pe_vector_res_2_meta:np.array,
        pe_vector_bead_2_res:np.array,
        meta_2_node_edge:torch.tensor,
        meta_2_node_vector:torch.tensor,
        attention_layer:int,
        e_hidden_dim: Union[int, list, tuple],
        z_dim:int,
        latent_dim: int,
        d_hidden_dim:Union[int, list, tuple],
        e_hidden_layers: int,
        d_hidden_layers: int,
    ):
        super().__init__()
        if encoder_cls == "MLP":
            self.encoder = VAEEncoder(in_dim, e_hidden_dim, z_dim, e_hidden_layers)
        elif encoder_cls == "GAT":
            self.encoder = GATEncoder(in_dim, e_hidden_dim, attention_layer, z_dim, e_hidden_layers)
        elif encoder_cls == "MS-GAT":
            self.encoder = MultiScaleGATEncoder(in_dim, e_hidden_dim, attention_layer, z_dim, e_hidden_layers)
        else:
            raise Exception()
        if decoder_cls == "MLP":
            self.decoder = Decoder(z_dim, d_hidden_dim, out_dim, d_hidden_layers)
        elif decoder_cls == "GNN_with_prior":
            self.decoder = EGNNDecoder(d_hidden_dim, d_hidden_layers)
        elif decoder_cls == "metaGNN":
            self.decoder = HierarchicalDeltaGNN_CG(in_dim = z_dim, d_hidden_dim = d_hidden_dim, latent_dim = latent_dim, sec_ids=sec_ids, beads_ids = beads_ids,meta_edge_index=meta_edge_index, 
                                                edge_dist=edge_dist, pe_vector_res_2_meta = pe_vector_res_2_meta, pe_vector_bead_2_res = pe_vector_bead_2_res,meta_2_node_edge = meta_2_node_edge, meta_2_node_vector = meta_2_node_vector,out_dim=out_dim)
        elif decoder_cls == "metaGNN_old":
            self.decoder = HierarchicalDeltaGNN_old(in_dim = z_dim, d_hidden_dim = d_hidden_dim, latent_dim = latent_dim, sec_ids=sec_ids, meta_edge_index=meta_edge_index, 
                                                edge_dist=edge_dist, pe_vector = pe_vector_res_2_meta, meta_2_node_edge = meta_2_node_edge, meta_2_node_vector = meta_2_node_vector,out_dim=out_dim)
        else:
            print(f"{decoder_cls} not in presets, you may set it manually later.")
            self.decoder: torch.nn.Module

    def encode(self,x):
        mean = self.encoder(x)
        return mean

    def forward(self, img):
        z = self.encode(img)
        out = self.decoder(z)
        return out, z
    
    def eval_z(self, z):
        out = self.decoder(z)
        return out