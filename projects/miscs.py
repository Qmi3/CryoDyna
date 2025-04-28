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

from cryostar.common.residue_constants import ca_ca
from cryostar.utils.misc import log_to_current
from cryostar.utils.ml_modules import VAEEncoder, EGNNDecoder, reparameterize,Decoder,  GATEncoder,HierarchicalDeltaGNN, MultiScaleGATEncoder
from cryostar.utils.ctf import parse_ctf_star
# from cryostar.openfold.utils.utils import positionalencoding1d,positionalencoding2d
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
    mask = np.asarray(freq < 1 / bandwidth, dtype=np.float32)
    return mask


def get_edges_batch(edges,edge_attr,n_nodes, batch_size):
    
    edges = torch.LongTensor(edges)
    rows, cols = [], []
    for i in range(batch_size):
        rows.append(edges[:,0] + n_nodes * i)
        cols.append(edges[:,1] + n_nodes * i)
    # edges = [torch.cat(rows), torch.cat(cols)]
    edges = torch.stack([torch.cat(rows), torch.cat(cols)], dim=-1)
    # edge_attr = torch.tensor(edge_attr).repeat(batch_size,1)
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


@lru_cache(maxsize=None)
def prepare_dynamic_loss(
        num_nodes: int,
        pair_index: torch.LongTensor,    # shape: (edge, 2)
        top_p_ratio: float,
    ):
    """
        The left side of pair_index should be sorted in the ascending order!
        [
            [0, _], [0, _], [0, _],
            [1, _], [1, _],
            [2, _], [2, _], [2, _], [2, _],
            ...
        ]
    """
    device = pair_index.device
    num_node_nbrs = [0 for _ in range(num_nodes)]
    left_nodes = pair_index[:, 0].tolist()
    for ele in left_nodes:
        num_node_nbrs[ele] += 1
    num_node_nbrs = torch.tensor(num_node_nbrs, device=device)
    reshape_indices = torch.zeros(num_nodes, max(num_node_nbrs), dtype=torch.long, device=device)
    reshape_valid_mask = torch.zeros(num_nodes, max(num_node_nbrs), dtype=torch.bool, device=device)
    reshape_top_p_mask = torch.zeros(num_nodes, max(num_node_nbrs), dtype=torch.bool, device=device)
    start_idx = 0
    for i in range(num_nodes):
        reshape_indices[i, :num_node_nbrs[i]] = start_idx + torch.arange(num_node_nbrs[i], device=device)
        reshape_valid_mask[i, :num_node_nbrs[i]] = True
        reshape_top_p_mask[i, :int(top_p_ratio * num_node_nbrs[i])] = True
        start_idx += num_node_nbrs[i]
    return num_node_nbrs, reshape_indices, reshape_valid_mask, reshape_top_p_mask


@lru_cache(maxsize=None)
def prepare_dynamic_intra_chain_loss(
    chain_id: tuple,                # shape: (node, ), converted from np.ndarray since it may be unhashable
    pair_index: torch.LongTensor,   # shape: (edge, 2)
):
    chain_id = np.array(chain_id)
    device = pair_index.device
    chain2idx = {}
    idx = 0
    for ele in set(chain_id):
        chain2idx[ele] = idx
        idx += 1

    chain_pairs = [[] for _ in range(len(chain2idx))]
    pair_index_np = pair_index.cpu().numpy()
    pair_chain_id = chain_id[pair_index_np]
    for pair_idx, pair in enumerate(pair_chain_id):
        if pair[0] == pair[1]:
            chain_pairs[chain2idx[pair[0]]].append(pair_idx)
    chain_pairs = [torch.tensor(ele, device=device) for ele in chain_pairs if len(ele) > 10]
    return chain_pairs


def calc_pair_dist_loss(pred_struc, pair_index, target_dist, type="vanilla", chain_id=None):
    bsz = pred_struc.shape[0]
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    if type == "vanilla":
        return F.mse_loss(pred_dist, target_dist.repeat(bsz, 1))
    elif "all-var-relax" in type:
        # optional value:
        #   all-var-relax@p0.99    keep bonds whose variance is the smallest 99%
        #   all-var-relax@q1.0     keep bonds whose variance >= 1.0
        if "@" in type:
            arg = type.split("@")[1]
            assert arg[0] in ["p", "q"]
            use_percentile = arg[0] == "p"
            loss_filter = float(arg[1:])
        else:
            use_percentile = True
            loss_filter = 0.99
        loss = F.mse_loss(pred_dist, target_dist.repeat(bsz, 1), reduction="none")
        loss_var = loss.var(0, keepdim=False).detach()
        # if "var-relax-ema" in type:
        #     other.running_variance = 0.9 * other.running_variance + 0.1 * loss_var
        #     loss_var = other.running_variance
        if np.random.rand() < 0.001:
            log_to_current("variance statistics:")
            q = [0.0, 0.9, 0.95, 0.97, 0.99, 0.999]
            v = torch.quantile(loss_var, torch.tensor(q, device=loss.device)).tolist()
            log_to_current("|".join([f" {q[i] * 100}%: {v[i]:.3f} " for i in range(len(q))]))
            p = [0.25, 1.0, 4.0, 16.0]
            v = [(loss_var > p[i]).sum() / len(loss_var) for i in range(len(p))]
            log_to_current("|".join([f" {p[i]}: {v[i] * 100:.1f}% " for i in range(len(p))]))
        if use_percentile:
            loss_ind = loss_var.sort(descending=False).indices
            loss = loss.index_select(1, loss_ind[:int(len(loss_var) * loss_filter)])
        else:
            loss_mask = loss_var < loss_filter
            loss = loss[loss_mask[None, :].repeat(bsz, 1)]
        avg_loss = loss.mean()
        return avg_loss
    elif "chain-var-relax" in type:
        if "@" in type:
            arg = type.split("@")[1]
            loss_filter = float(arg[1:])
        else:
            loss_filter = 0.95
        loss = F.mse_loss(pred_dist, target_dist.repeat(bsz, 1), reduction="none")
        chain_pairs = prepare_dynamic_intra_chain_loss(tuple(chain_id), pair_index)
        chain_losses = []
        for i in range(len(chain_pairs)):
            chain_loss = loss.index_select(1, chain_pairs[i])
            chain_loss_var = chain_loss.var(0, keepdim=False).detach()
            chain_loss_ind = chain_loss_var.sort(descending=False).indices
            chain_loss = chain_loss.index_select(1, chain_loss_ind[:int(len(chain_loss_var) * loss_filter)])
            chain_losses.append(chain_loss)
        loss = torch.cat(chain_losses, 1)
        avg_loss = loss.mean()
        return avg_loss
    elif type == "inverse":
        target_dist = target_dist.repeat(bsz, 1)
        loss = F.mse_loss(pred_dist, target_dist, reduction="none")
        lt6_loss = (loss[target_dist <= 6]).sum()
        gt6_loss = loss[target_dist > 6]
        gt6_weight = 1 / (target_dist[target_dist > 6].detach() - 5)
        gt6_loss = (gt6_loss * gt6_weight).sum()
        total_loss = lt6_loss + gt6_loss
        avg_loss = total_loss / target_dist.numel()
        return avg_loss
    elif "dynamic" in type:
        if "@" in type:
            ratio = float(type.split("@")[1])
        else:
            ratio = 0.85
        num_nodes = pred_struc.shape[1]
        num_node_nbrs, reshape_indices, reshape_valid_mask, reshape_top_p_mask = prepare_dynamic_loss(
            num_nodes, pair_index, ratio)
        dist_mse = (pred_dist - target_dist)**2  # bsz x num_nodes
        dist_mse = dist_mse.index_select(1, reshape_indices.reshape(-1))  # bsz x (num_nodes * max_node_nbr)
        dist_mse = dist_mse.reshape(bsz, num_nodes, num_node_nbrs.max())
        dist_mse = dist_mse.masked_fill(~reshape_valid_mask[None, ...], 10000.)
        dist_mse = dist_mse.sort(descending=False, dim=2).values  # bsz x num_nodes x max_node_nbr
        batch_mask = einops.repeat(reshape_top_p_mask, "num_nodes max_node_nbr -> bsz num_nodes max_node_nbr", bsz=bsz)
        avg_loss = dist_mse[batch_mask].sum() / batch_mask.sum()
        return avg_loss
    elif type == "90p":
        target_dist = target_dist.repeat(bsz, 1)
        loss = F.mse_loss(pred_dist, target_dist, reduction="none")
        mask = torch.le(loss, torch.quantile(loss, 0.9, dim=1, keepdim=True))
        avg_loss = loss[mask].sum() / mask.float().sum()
        return avg_loss
    else:
        raise NotImplementedError


class VAE(nn.Module):

    def __init__(
        self,
        encoder_cls: str,
        decoder_cls: str,
        in_dim: int,
        # edge_index: np.array,
        # edge_dist: np.array,
        # edge_bond: np.array,
        # pos:np.array,
        # pae:np.array,
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
        # d_e_hidden_dim:int,
        # d_hidden_dim: Union[int, list, tuple],
        # out_dim: int,
        e_hidden_layers: int,
        d_hidden_layers: int,
        # pe_dim:int
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
        # self.edges = edge_index
        # #self.edge_dist = edge_dists
        # self.edge_attr = edge_bond
        # self.pe_dim = pe_dim
        # self.mapping_adapter = Adapter(latend_dim,out_dim)
        # self.mapping_layer = nn.Sequential(nn.Linear(latent_dim , points_num * d_hidden_dim),nn.SiLU())
        # self.pos = pos
        # self.pae = torch.from_numpy(pae)
        if decoder_cls == "MLP":
            # self.decoder = Decoder(latent_dim+pe_dim, d_hidden_dim, out_dim, d_hidden_layers)
            self.decoder = Decoder(z_dim, d_hidden_dim, out_dim, d_hidden_layers)
        elif decoder_cls == "GNN_with_prior":
            self.decoder = EGNNDecoder(d_hidden_dim, d_hidden_layers)
        elif decoder_cls == "metaGNN":
            self.decoder = HierarchicalDeltaGNN(in_dim = z_dim, d_hidden_dim = d_hidden_dim, latent_dim = latent_dim, sec_ids=sec_ids, meta_edge_index=meta_edge_index, 
                                                edge_dist=edge_dist, pe_vector = pe_vector,meta_2_node_edge = meta_2_node_edge, meta_2_node_vector = meta_2_node_vector,out_dim=out_dim)
        else:
            print(f"{decoder_cls} not in presets, you may set it manually later.")
            self.decoder: torch.nn.Module
        # self.layer_norm = nn.LayerNorm(384)

    def encode(self,x):
        mean = self.encoder(x)
        return mean
        # mean, log_var = self.encoder(x)
        # return mean, log_var

    # def mapping(self,z):
    #     h = self.mapping_layer(z)
    #     return h
    # def latent_encoding(self,z,L):
    #     bz = z.shape[0]
    #     pe = positionalencoding2d(self.pe_dim,L,L)
    #     z = z.repeat_interleave(L*L,dim=0)
    #     pe = pe.view(L*L,-1).repeat(bz,1).to(z.device)
    #     z_new = torch.cat([z,pe],-1)
    #     return z_new
    def forward(self, img):
        z = self.encode(img)
        # z = reparameterize(mean, log_var)
        # out = self.decoder(z, self.pos, self.pae)
        out = self.decoder(z)
        return out, z
    
    def eval_z(self, z):
        # out = self.decoder(z, self.pos, self.pae)
        out = self.decoder(z)
        return out
    
    # def forward(self, img):
    #     L = xyz_0.shape[0]
    #     mean, log_var = self.encode(img)
    #     z = reparameterize(mean, log_var)
    #     h = self.mapping(z)
    #     h = h.view(img.shape[0]* L,-1)
    #     xyz_0 = xyz_0.repeat(img.shape[0],1)
    #     # extract 64 edges from reference structure
    #     edges_batch,edge_attr_batch = get_edges_batch(self.edges, self.edge_attr,L,img.shape[0])
    #     # edges_batch,edge_attr_batch,edge_bond_batch = get_edges_batch(self.edges, self.edge_attr,self.edge_bond ,L,img.shape[0])
    #     # encode the edge features
    #     # print(edges_batch[0].shape,edges_batch[1].shape,edge_attr_batch.shape,edge_bond_batch.shape)
    #     edges_batch,edge_attr_batch = edges_batch.to(img.device),edge_attr_batch.to(img.device)
    #     out = self.decoder(h,xyz_0,edges_batch,edge_attr_batch)
    #     out = out.view(img.shape[0],L,-1)
    #     return out, mean, log_var
    
    # def eval_z(self, z , xyz_0):
    #     L = xyz_0.shape[0]
    #     hidden = self.mapping(z)
    #     hidden = hidden.view(z.shape[0]*L,-1)
    #     xyz_0 = xyz_0.repeat(z.shape[0],1)
    #     # extract 64 edges from reference structure
    #     # edges_batch,edge_attr_batch = get_edges_batch(self.edges, self.edge_attr,self.edge_bond ,L,z.shape[0])
    #     # encode the edge features
    #     edges_batch,edge_attr_batch = get_edges_batch(self.edges, self.edge_attr,L,z.shape[0])
    #     edges_batch,edge_attr_batch = edges_batch.to(z.device),edge_attr_batch.to(z.device)
    #     # import pdb;pdb.set_trace()
    #     out = self.decoder(hidden,xyz_0,edges_batch,edge_attr_batch)
    #     out = out.view(z.shape[0],L,-1)
    #     return out

