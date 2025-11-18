# Ref: https://github.com/sokrypton/ColabDesign.git

import torch
import numpy as np
import pickle
from Bio import SeqIO
from cryostar.openfold.utils.residue_constants import restype_3to1
from Bio import Align
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import fsolve

def get_rmsd_loss(true_protein, pred_CA, L=None, include_L=True, copies=1):
  B = pred_CA.size()[0]
  true_CA = true_protein.repeat(B, 1, 1)
  weights=None
  # true_CA = torch.tensor(true_protein.atom_positions[:, 1, :], device=device).repeat(B, 1, 1)
  # pred = outputs["structure_module"]["final_atom_positions"][:,1]
  # weights = torch.tensor(true_protein.atom_mask[:, 1], device=device)
  return _get_rmsd_loss(pred_CA, true_CA, weights=weights, L=L, include_L=include_L, copies=copies)

def _get_rmsd_loss(true, pred, weights=None, L=None, include_L=True, copies=1):
  '''
  get rmsd + alignment function
  align based on the first L positions, computed weighted rmsd using all 
  positions (if include_L=True) or remaining positions (if include_L=False).
  '''
  # normalize weights
  length = true.shape[-2]
  if weights is None:
    weights = (torch.ones(length, device=true.device)/length)[...,None]
  else:
    weights = (weights/(weights.sum(-1,keepdims=True) + 1e-8))[...,None]

  # determine alignment [L]ength and remaining [l]ength
  if copies > 1:
    if L is None:
      L = iL = length // copies; C = copies-1
    else:
      (iL,C) = ((length-L) // copies, copies)
  else:
    (L,iL,C) = (length,0,0) if L is None else (L,length-L,1)

  # slice inputs
  if iL == 0:
    (T,P,W) = (true,pred,weights)
  else:
    (T,P,W) = (x[...,:L,:] for x in (true,pred,weights))
    (iT,iP,iW) = (x[...,L:,:] for x in (true,pred,weights))

  # get alignment and rmsd functions
  (T_mu,P_mu) = ((x*W).sum(-2,keepdims=True)/W.sum((-1,-2)) for x in (T,P)) # 计算质心
  aln = _kabsch((P-P_mu)*W, T-T_mu)   
  
  align_fn = lambda x: (x - P_mu) @ aln + T_mu
  msd_fn = lambda t,p,w: (w*torch.square(align_fn(p)-t)).sum((-1,-2))
  # compute rmsd
  if iL == 0:
    msd = msd_fn(true,pred,weights)
  elif C > 1:
    # all vs all alignment of remaining, get min RMSD
    iT = iT.reshape(-1,C,1,iL,3).swapaxes(0,-3)
    iP = iP.reshape(-1,1,C,iL,3).swapaxes(0,-3)
    imsd = msd_fn(iT, iP, iW.reshape(-1,C,1,iL,1).swapaxes(0,-3))
    imsd = (imsd.min(0).sum(0) + imsd.min(1).sum(0)) / 2 
    imsd = imsd.reshape(torch.broadcast_shapes(true.shape[:-2],pred.shape[:-2]))
    msd = (imsd + msd_fn(T,P,W)) if include_L else (imsd/iW.sum((-1,-2)))
  else:
    msd = msd_fn(true,pred,weights) if include_L else (msd_fn(iT,iP,iW)/iW.sum((-1,-2)))
  rmsd = torch.sqrt(msd + 1e-8)

  return {"rmsd":rmsd, "align":align_fn, "rot":aln.float(), "trans": (T_mu-P_mu @ aln)[:, 0, :].float()}

def _kabsch(a, b, return_v=False):
    '''get alignment matrix for two sets of coordinates using PyTorch'''
    ab = torch.matmul(a.transpose(-1, -2), b) 
    u, s, vh = torch.linalg.svd(ab, full_matrices=False)
    flip = torch.det(torch.matmul(u, vh)) < 0
    u_ = torch.where(flip.unsqueeze(-1), -u[..., -1], u[..., -1]).unsqueeze(-1) 
    u = torch.cat((u[..., :-1], u_), dim=-1) if flip.any() else u 
    return u if return_v else torch.matmul(u, vh)


def gaussian_fit_and_call_threshold(data):
  gmm = GaussianMixture(n_components=3,  random_state=0)
  gmm.fit(data.reshape(-1, 1))
  weights = gmm.weights_
  mu1, mu2, mu3 = np.sort(gmm.means_.flatten())
  sigma1, sigma2, sigma3 = np.sqrt(gmm.covariances_).flatten()
  w1, w2, w3 = weights[np.argsort(gmm.means_.flatten())]
  def gaussian1(x):
      return w1 * norm.pdf(x, mu1, sigma1)
  def gaussian2(x):
      return w2 * norm.pdf(x, mu2, sigma2)
  def gaussian3(x):
      return w3 * norm.pdf(x, mu3, sigma3)
  # **计算交汇点**
  def intersection(x):
      return gaussian2(x) - gaussian3(x)
  x0 = [(mu2 + mu3) / 2]
  threshold = fsolve(intersection, x0)[0]
  return threshold

def call_relaxed_edge(config,meta):
  intra_chain_mask = (meta.chain_id[None,:] == meta.chain_id[:,None])
  fold_feature = pickle.load(open(config.extra_input_data_attr.fold_feature_path,'rb'))
  pae_intra = fold_feature['predicted_aligned_error']
  plddt = fold_feature['plddt']
  seqs = list(SeqIO.parse(config.extra_input_data_attr.pred_sequence,'fasta'))
  fasta_full = ''.join([str(seq.seq) for seq in seqs])
  pdb_fasta = ''.join([restype_3to1[res] for res in meta.res_name])
  aligner = Align.PairwiseAligner()
  aligner.target_gap_score = -100
  aligner.query_open_gap_score = -100
  alignments = aligner.align(fasta_full,pdb_fasta)
  sel = []
  for index,aa in enumerate(alignments[0][1]):
    if aa != '-':
      sel.append(index)
  sel = np.array(sel)
  pae_intra =  pae_intra[sel[:,None],sel[None,:]]
  plddt = plddt[sel]
  pae_intra[~ intra_chain_mask] = 0
  pae_intra[plddt < 75,:] = 0
  pae_intra[:,plddt< 75] = 0
  pae_threshold = gaussian_fit_and_call_threshold(pae_intra[pae_intra !=0 ])
  relaxed_edges = np.where((np.triu(pae_intra,1) >= pae_threshold))
  non_relaxed_edges = np.where((np.triu(pae_intra,1) < pae_threshold) & (np.triu(intra_chain_mask,1) == True))
  return np.array(relaxed_edges).T,np.array(non_relaxed_edges).T


def torch_interp(x, xp, fp):
    """PyTorch 实现的 1D 线性插值（类似 numpy.interp）"""
    indices = torch.searchsorted(xp, x, right=True)
    indices = torch.clamp(indices, 1, len(xp) - 1)  # 防止超出范围

    x0, x1 = xp[indices - 1], xp[indices]
    f0, f1 = fp[indices - 1], fp[indices]

    slope = (f1 - f0) / (x1 - x0+ 1e-9)
    return f0 + slope * (x - x0)

def wasserstein_distance(X, Y):
    X_sorted = torch.sort(X)[0]  # 对 X 进行排序
    Y_sorted = torch.sort(Y)[0]  # 对 Y 进行排序

    m, n = len(X), len(Y)

    # 计算经验 CDF
    F_X = torch.linspace(1/m, 1, m, device=X.device)
    F_Y = torch.linspace(1/n, 1, n, device=Y.device)

    # 统一所有数据点
    all_sorted = torch.cat((X_sorted, Y_sorted)).unique()

    # 进行 CDF 插值
    F_X_interp = torch_interp(all_sorted, X_sorted, F_X)
    F_Y_interp = torch_interp(all_sorted, Y_sorted, F_Y)

    # 计算 Wasserstein-1 距离
    distance = torch.sum(torch.abs(F_X_interp - F_Y_interp) * torch.diff(torch.cat([torch.tensor([0.], device=X.device), all_sorted])))

    return distance

from pymol import cmd
backbone_atoms = "N+CA+C+O"
def get_rmsd(pdb1, pdb2, cycles=0):
    cmd.load(pdb1, 'obj1')
    cmd.load(pdb2, 'obj2')
    align = cmd.align('obj1','obj2', cycles=cycles)
    cmd.delete('all')
    return align[0]

def get_Ca_rmsd(pdb1, pdb2, cycles=0):
    cmd.load(pdb1, 'obj1')
    cmd.load(pdb2, 'obj2')
    align = cmd.align("obj1 and (name CA or name C1')','obj2 and (name CA or name C1')", cycles=cycles)
    cmd.delete('all')
    return align[0]
  
def get_backbone_rmsd(pdb1, pdb2, cycles=0):
    cmd.load(pdb1, 'obj1')
    cmd.load(pdb2, 'obj2')
    try:
      align = cmd.align(f'obj1 and name {backbone_atoms} and polymer',
                        f'obj2 and name {backbone_atoms} and polymer', cycles=cycles)
      cmd.delete('all')
      return align[0]
    except:
      cmd.delete('all')
      return 0
# def sigmoid(x):
#   x_ravel = x.ravel()  # 将numpy数组展平
#   length = len(x_ravel)
#   y = []
#   for index in range(length):
#       if x_ravel[index] >= 0:
#           y.append(1.0 / (1 + np.exp(-x_ravel[index])))
#       else:
#           y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
#   return np.array(y).reshape(x.shape)

# def norm_and_extract(config,meta):
#   fold_feature = pickle.load(open(config.extra_input_data_attr.fold_feature_path,'rb'))
#   # raw_distogram = 1/(1+np.exp(-1*fold_feature['distogram_logits']))
#   if 'distogram' in fold_feature.keys():
#     raw_distogram = sigmoid(fold_feature['distogram']['logits'])
#   else:
#     raw_distogram = sigmoid(fold_feature['distogram_logits'])
#   dist_norm1 = raw_distogram/np.sum(raw_distogram,axis=(-1,-2))[:,None,None]
#   dist_norm2 = dist_norm1/np.sum(dist_norm1,axis=-1)[:,:,None]
#   seqs = list(SeqIO.parse(config.extra_input_data_attr.pred_sequence,'fasta'))
#   fasta_full = ''.join([str(seq.seq) for seq in seqs])
#   pdb_fasta = ''.join([restype_3to1[res] for res in meta.res_name])
#   aligner = Align.PairwiseAligner()
#   aligner.target_gap_score = -100
#   aligner.query_open_gap_score = -100
#   alignments = aligner.align(fasta_full,pdb_fasta)
#   sel = []
#   for index,aa in enumerate(alignments[0][1]):
#     if aa != '-':
#       sel.append(index)
#   sel = np.array(sel)
#   dist_norm3 = dist_norm2[sel[:,None],sel[None,:]]
#   return dist_norm3
