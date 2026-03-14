import os
import sys
sys.path.insert(0,'/lustre/grp/gyqlab/lism/CryoDyna')
from mmengine import Config
import torch
import numpy as np
import pickle
from cryodyna.gmm.deformer import E3Deformer, NMADeformer
from cryodyna.utils.align import get_rmsd_loss
from cryodyna.utils.polymer import Polymer
from miscs import VAE
from cryodyna.martini.IO import streamTag, pdbFrame, pdbChains, residues, Chain, pdbOut, pdbBoxString
# work_dir = '/lustre/grp/gyqlab/lism/cryostar/projects/1ake_cg_all_morse0.1/atom_1ake_9/'
# num_step = '0079_0062480'
# work_dir = '/lustre/grp/gyqlab/lism/cryostar/projects/1ake_cg_all_morse0.1_improper1.0_dist1.0_3.56_bb_filter/atom_1ake'
# work_dir = "/lustre/grp/gyqlab/lism/CryoDyna/spike_MD_cg/atom_spike_MD_updated"
work_dir = sys.argv[1]
# num_step = '0079_0031200'
for num_step in os.listdir(work_dir):
    if num_step == '0000_0000000':
        continue
    if not os.path.exists(f'{work_dir}/{num_step}/ckpt.pt'):
        continue
    save_dir = f'{work_dir}/{num_step}/sampled_pdb'
    os.makedirs(save_dir, exist_ok=True)
    cfg = Config.fromfile(f'{work_dir}/config.py')
    device = 'cuda:0'

    inStream = streamTag(cfg.dataset_attr.ref_pdb_path)
    fileType = next(inStream)
    if fileType != "PDB":
        raise NotImplementedError
    title, atoms, box = pdbFrame(inStream)
    chains = [Chain([i for i in residues(chain)]) for chain in pdbChains(atoms)]
    n = 1
    for chain in chains:
        n += 1
    # Check all chains
    keep = []
    for chain in chains:
        if chain.type() == "Water":
            logging.info("Removing %d water molecules (chain %s)." % (len(chain), chain.id))
        elif chain.type() in ("Protein", "Nucleic"):
            keep.append(chain)
        # This is currently not active:
        elif options['RetainHETATM']:
            keep.append(chain)
        else:
            logging.info("Removing HETATM chain %s consisting of %d residues." % (chain.id, len(chain)))
    chains = keep
    # Get the total length of the sequence
    seqlength = sum([len(chain) for chain in chains])

    coarseGrained = []
    electrons = []
    sizes = []
    coords = []
    num_beads = []
    chain_id = []
    res_id = []
    for chain in chains:
        coarseGrained_, electrons_ = chain.cg(com=True)
        if coarseGrained_:
            coarseGrained.extend(coarseGrained_)
            electrons.extend(electrons_)
            num_beads.append(len(coarseGrained_))
            for name, resn, resi, chain, x, y, z in coarseGrained_:
                insc = resi >> 20
                resi = resi-(insc << 20)
                coords.append([x, y, z])
                chain_id.append(chain)
                res_id.append(resi)
        else:
            raise ValueError("No mapping for coarse graining chain %s (%s); chain is skipped." % (chain.id, chain.type()))
    template_pdb = coarseGrained
    # ref + cg
    ref_centers = torch.tensor(coords).float().to(device)
    ref_amps = torch.tensor(electrons).float().to(device)

    num_pts = ref_centers.shape[0]

    # model
    if cfg.model.input_space == "fourier":
        in_dim = 2 * cfg.data_process.down_side_shape ** 2
    elif cfg.model.input_space == "real":
        in_dim = cfg.data_process.down_side_shape ** 2
    else:
        raise NotImplementedError
    struc_feature = pickle.load(open(f'{work_dir}/struc_fea.pkl','rb'))
    sec_ids = struc_feature['sec_ids']
    res_id_non_rb = struc_feature['res_id_non_rb']
    meta_edge_index = struc_feature['meta_edge_index']
    edge_dist = struc_feature['edge_dist']
    pe_vector_res_2_meta = struc_feature['pe_vector_res_2_meta']
    pe_vector_bead_2_res = struc_feature['pe_vector_bead_2_res']
    meta_2_node_edge = struc_feature['meta_2_node_edge']
    meta_2_node_vector = struc_feature['meta_2_node_vector']
    nma_modes = None
    if (hasattr(cfg.extra_input_data_attr, "nma_path") and cfg.extra_input_data_attr.nma_path not in ["", None]):
        nma_modes = torch.tensor(np.load(cfg.extra_input_data_attr.nma_path), dtype=torch.float32).to(device)
    model = VAE(in_dim=in_dim,
            out_dim=num_pts * 3 if nma_modes is None else 6 + nma_modes.shape[1],
            sec_ids = torch.from_numpy(np.array(sec_ids)).long(),
            beads_ids = torch.from_numpy(res_id_non_rb),
            meta_edge_index = meta_edge_index.long(),
            edge_dist = torch.from_numpy(edge_dist).float(),
            pe_vector_res_2_meta = torch.stack(pe_vector_res_2_meta,dim=0),
            pe_vector_bead_2_res = torch.stack(pe_vector_bead_2_res,dim=0),
            meta_2_node_edge = meta_2_node_edge.long(),
            meta_2_node_vector = meta_2_node_vector,
            **cfg.model.model_cfg)

    if nma_modes is None:
        deformer = E3Deformer()
    else:
        deformer = NMADeformer(nma_modes)
    z_list = np.load(f'{work_dir}/{num_step}/z.npy')
    print(f'{work_dir}/{num_step}/z.npy')
    z = torch.from_numpy(z_list)
    # sampled_z_index = np.arange(0,50000,100)
    sampled_z_index = np.arange(0,100000,100)
    sampled_z = z[sampled_z_index]
    weights = torch.load(f'{work_dir}/{num_step}/ckpt.pt')
    model.load_state_dict(weights['model'])
    model.eval()
    model.to(device)
    with torch.no_grad():
        sampled_z = sampled_z.float().to(device)
        pred_deformation = model.decoder(sampled_z)
        pred_struc = deformer.transform(pred_deformation, ref_centers)
        pred_struc = pred_struc.squeeze(0)

    ref_cg = template_pdb.copy()
    for i, z_idx in enumerate(sampled_z_index):
        cgOutPDB = open(f'{save_dir}/sampled_pdb_{z_idx}.pdb', "w")
        atid = 1
        cgOutPDB.write(f"MODEL {z_idx+1}\n")
        cgOutPDB.write(title)
        cgOutPDB.write(pdbBoxString(box))
        tmp_struc = pred_struc[i].cpu().numpy()
        for j, (name, resn, resi, chain, x_ref, y_ref, z_ref) in enumerate(ref_cg):
            x, y, z = tmp_struc[j]
            cgOutPDB.write(pdbOut((name, resn[:3], resi, chain, x, y, z),i=atid))
            atid += 1
        cgOutPDB.write("ENDMDL\n")
        cgOutPDB.close()