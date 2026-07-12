import argparse
from cryodyna.utils.pdb_tools import bt_read_pdb
from cryodyna.gmm.gmm import EMAN2Grid, Gaussian, canonical_density
from cryodyna.utils.polymer import Polymer
from cryodyna.utils.mrc_tools import save_mrc
import torch


def pdb_to_mrc(pdb_path, out_path, shape=128, apix=None, sigma=2.0):
    """Convert PDB to MRC density map."""
    apix = apix or 1.4 * 380 / shape
    
    atom_arr = bt_read_pdb(pdb_path)[0]
    meta = Polymer.from_atom_arr(atom_arr)
    
    centers = torch.from_numpy(meta.coord).float()
    amps = torch.from_numpy(meta.num_electron).float()
    sigmas = torch.ones_like(amps) * sigma
    
    grid = EMAN2Grid(side_shape=shape, voxel_size=apix)
    vol = canonical_density(
        Gaussian(mus=centers, sigmas=sigmas, amplitudes=amps),
        grid.line()
    )
    vol = vol.permute(2, 1, 0).cpu().numpy()
    
    save_mrc(vol, out_path, apix, -shape // 2 * apix)
    print(f"✅ MRC saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDB file to MRC density map")
    parser.add_argument("-i", "--input", required=True, help="Input PDB file path")
    parser.add_argument("-o", "--output", required=True, help="Output MRC file path")
    parser.add_argument("-s", "--shape", type=int, default=128, help="Target side shape (default: 128)")
    parser.add_argument("-a", "--apix", type=float, default=None, help="Voxel size in Å/pixel (default: auto-calculated)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma (default: 2.0)")
    
    args = parser.parse_args()
    pdb_to_mrc(args.input, args.output, args.shape, args.apix, args.sigma)