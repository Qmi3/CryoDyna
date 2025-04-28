import pickle
import os
import numpy as np
import math
import torch
from cryostar.openfold.utils.import_weights import import_jax_weights_
from cryostar.openfold.utils import residue_constants, protein
from cryostar.openfold.fold_trunk import AlphaFold_Structure_Module
from cryostar.openfold.utils.tensor_utils import tensor_tree_map
import torch.nn as nn


def prep_input_feature(protein_feature_path ,fold_feature_path):
    out_keys=['single','pair','final_atom_mask']
    feat_keys = ['aatype','residx_atom37_to_atom14','atom37_atom_exists','seq_mask','residue_index','asym_id','entity_id','atom14_atom_exists','residx_atom37_to_atom14']
    fold_feature = {}
    feat_dict = pickle.load(open(protein_feature_path,'rb'))
    feat = {k:v for k,v in feat_dict.items() if k in feat_keys }
    out_dict = pickle.load(open(fold_feature_path,'rb'))
    out = {k:v for k,v in out_dict.items() if k in out_keys }
    fold_feature.update(out)
    fold_feature.update(feat)
    return fold_feature

def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]
    
# def load_models_from_command_line(config, model_device, jax_param_path):
#     for path in jax_param_path.split(","):
#         model_basename = get_model_basename(path)
#         model_version = "_".join(model_basename.split("_")[1:])
#         model = AlphaFold_Structure_Module(config)
#         model = model.eval()
#         import_jax_weights_(
#             model, path, partial=True, version=model_version
#         )
#         model = model.to(model_device)
#         yield model

def load_models_from_command_line(config,jax_param_path,fold_feature,num_ipa):
    for path in jax_param_path.split(","):
        model_basename = get_model_basename(path)
        model_version = "_".join(model_basename.split("_")[1:])
        # config.globals.chunk_size = 384
        config.model.structure_module.no_blocks = num_ipa
        model = AlphaFold_Structure_Module(config,fold_feature)
        # model = model.to(device)
        model = model.eval()
        import_jax_weights_(
            model, path, partial=True, version=model_version
        )
        # model.freeze()
        # model = model.to(model_device)
        # total_num = 0
        for p in model.parameters():
            p.requires_grad = False
        #     total_num += p.numel()
        # print(total_num)
        # import pdb;pdb.set_trace()
        return model

        
def prep_output(out, batch, multimer_ri_gap, subtract_plddt):
    plddt = out["plddt"]

    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if subtract_plddt:
        plddt_b_factors = 100 - plddt_b_factors
        
    ri = batch["residue_index"]
    chain_index = (ri - np.arange(ri.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if c != cur_chain:
            cur_chain = c
            prev_chain_max = i + cur_chain * multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False
    )

    return unrelaxed_protein

def prep_structure(feature_dict,out,batch_size):
    # feature_dict = tensor_tree_map(lambda x: np.array(x.detach().cpu()),feature_dict)
    out = tensor_tree_map(lambda x: np.array(x.detach().cpu()), out)
    multimer_ri_gap = 200
    subtract_plddt = True
    # feature_dict_ = {k:v[0] for k,v in feature_dict.items()}
    unrelaxed_proteins = []
    for bz in range(batch_size):
        out_ = {k:v[bz] for k,v in out.items() if k in ['final_atom_mask','final_atom_positions','plddt']}
        out_['final_atom_mask'] = feature_dict['final_atom_mask']
        unrelaxed_protein = prep_output(out_,feature_dict,multimer_ri_gap,subtract_plddt)
        # unrelaxed_file_suffix = f"_unrelaxed_{bz}.pdb"
        # unrelaxed_output_path = os.path.join(
        # pdb_path, f'test_{unrelaxed_file_suffix}'
        # )
        # with open(unrelaxed_output_path, 'w') as fp:
        #     fp.write(protein.to_pdb(unrelaxed_protein))
        unrelaxed_proteins.append(unrelaxed_protein)
    return unrelaxed_proteins

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

def positionalencoding1d(n_pos, dim):
  assert dim % 2 == 0, "wrong dim"
  n_pos_vec = torch.arange(n_pos, dtype=torch.float)
  position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float)

  omega = torch.arange(dim//2, dtype=torch.float)
  omega /= dim/2.
  omega = 1./(10000**omega)

  sita = n_pos_vec[:,None] @ omega[None,:]
  emb_sin = torch.sin(sita)
  emb_cos = torch.cos(sita)

  position_embedding[:,0::2] = emb_sin
  position_embedding[:,1::2] = emb_cos

  return position_embedding

def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()

def positionalencoding_af2learn(batch,device,d_model):
    pos = batch["residue_index"]
    asym_id = batch["asym_id"]
    max_relative_idx = 32
    max_relative_chain = 2
    asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
    offset = pos[..., None] - pos[..., None, :]
    clipped_offset = torch.clamp(
        offset + max_relative_idx, 0, 2 * max_relative_idx
    )
    rel_feats = []
    no_bins = (2 * max_relative_idx + 2 +1 + 2 * max_relative_chain + 2)
    linear_relpos = nn.Linear(no_bins,d_model)
    linear_relpos.to(device)
    final_offset = torch.where(
        asym_id_same, 
        clipped_offset,
        (2 * max_relative_idx + 1) * 
        torch.ones_like(clipped_offset))
    boundaries = torch.arange(
        start=0, end=2 * max_relative_idx + 2, device=final_offset.device)
    rel_pos = one_hot(
        final_offset,
        boundaries,)

    rel_feats.append(rel_pos)

    entity_id = batch["entity_id"]
    entity_id_same = (entity_id[..., None] == entity_id[..., None, :])
    rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))

    sym_id = batch["sym_id"]
    rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

    max_rel_chain = max_relative_chain
    clipped_rel_chain = torch.clamp(
        rel_sym_id + max_rel_chain,
        0,
        2 * max_rel_chain,
    )
    final_rel_chain = torch.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_rel_chain + 1) *
        torch.ones_like(clipped_rel_chain))
    boundaries = torch.arange(
        start=0, end=2 * max_rel_chain + 2, device=final_rel_chain.device)
    rel_chain = one_hot(
        final_rel_chain,
        boundaries,)
    rel_feats.append(rel_chain)
    rel_feat = torch.cat(rel_feats, dim=-1).to(
        linear_relpos.weight.dtype)
    return linear_relpos(rel_feat)
