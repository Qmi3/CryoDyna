# All following functions use two prefix to avoid naming conflicts:
# bp: Biopython, bt: biotite
import os.path as osp
from itertools import groupby
from pathlib import Path
from pprint import pprint
from typing import Iterable, Union
import torch
import biotite
import biotite.application.dssp as bt_dssp
import biotite.sequence as bt_seq
import biotite.sequence.graphics as graphics
import biotite.structure as bt_struc
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBIO, Chain, Entity, Model, PDBParser, Structure, MMCIFParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.parse_pdb_header import parse_pdb_header
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, get_structure
from matplotlib.patches import Rectangle

import Bio
if Bio.__version__ > "1.79":
    from Bio.PDB.Polypeptide import protein_letters_3to1
else:
    from Bio.PDB.Polypeptide import three_to_one as protein_letters_3to1


def _get_file_ext(file_path: Union[str, Path]):
    if isinstance(file_path, Path):
        return file_path.suffix
    else:
        return osp.splitext(file_path)[1]


# --------
# IO
# Some functions for read and write pdb files
def bp_read_pdb(file_path: Union[str, Path]):
    """Read pdb file by Biopython

    Parameters
    ----------
    file_path: pdb file path

    Returns
    -------
    structure: Biopython structure entity

    """
    file_ext = _get_file_ext(file_path)
    if file_ext == ".pdb":
        parser = PDBParser(PERMISSIVE=True)
    elif file_ext == ".cif":
        parser = MMCIFParser()
    else:
        raise NotImplementedError("Only support .pdb, .cif extension.")
    structure = parser.get_structure('tmp', file_path)
    return structure


def bp_save_pdb(file_path: Union[str, Path], structure: Entity):
    """Save Biopython structure to file

    Parameters
    ----------
    file_path: save file path
    structure: Biopython structure entity

    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(file_path)


def bt_read_pdb(file_path: Union[str, Path]):
    """Read pdb file by biotite, return all models as AtomArrayStack

    Parameters
    ----------
    file_path: pdb file path

    Returns
    -------
    atom_arr_stack: biotite AtomArrayStack containing all models

    """
    file_ext = _get_file_ext(file_path)
    if file_ext == ".pdb":
        f = PDBFile.read(file_path)
        atom_arr_stack = f.get_structure()
    elif file_ext == ".cif":
        f = PDBxFile.read(file_path)
        atom_arr_stack = get_structure(f)
    else:
        raise NotImplementedError("Only support .pdb, .cif extension.")
    return atom_arr_stack


def bt_save_pdb(file_path: Union[str, Path], array: Union[AtomArray, AtomArrayStack], **kwargs):
    """Save biotite AtomArray or AtomArrayStack to pdb file

    Parameters
    ----------
    file_path: save file path
    array: the structure to be saved
    kwargs: additional parameters to be passed, always empty

    """
    bt_struc.io.save_structure(file_path, array, **kwargs)


def print_pdb_header(file_path):
    """Print PDB file header

    Parameters
    ----------
    file_path: pdb file path

    Examples
    --------
    >>> print_pdb_header("5ni1.pdb")
    {'name': 'cryoem structure of haemoglobin at 3.2 a determined with the volta '
         'phase plate',
     'head': 'oxygen transport',
     ...}

    """
    pprint(parse_pdb_header(file_path))


def load_ca_coord(file_path):
    """Load CA (Carbon Alpha) coordinates from pdb file

    Parameters
    ----------
    file_path: pdb file path

    Returns
    -------
    coord (np.ndarray): [N, 3]

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import load_ca_coord
    >>> atom_arr = load_ca_coord(pdb_5ni1a_path())
    """
    atom_arr_stack = bt_read_pdb(file_path)
    if atom_arr_stack.stack_depth() > 1:
        print("Source pdb file contains models larger than 1, select the 1st.")
    atom_arr = atom_arr_stack[0]
    # filter backbone
    atom_arr = atom_arr[bt_struc.filter_peptide_backbone(atom_arr)]
    atom_arr = atom_arr[atom_arr.atom_name == "CA"]
    return atom_arr.coord


# --------
# PDB to Sequence
# Convert PDB data to sequence (VLS...)
def bp_chain2seq(chain: Chain):
    """Convert Biopython Chain to amino acid sequence

    Parameters
    ----------
    chain: Biopython Chain

    Returns
    -------
    AA type sequence string

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bp_read_pdb, bp_chain2seq
    >>> structure = bp_read_pdb(pdb_5ni1a_path())
    >>> bp_chain2seq(structure[0]['A'])
    'VLSPADKTNVKAAWGKVGAHAGE...'

    """
    return "".join([protein_letters_3to1[r.resname] for r in chain.get_residues()])


def bp_model2seq(model: Model, chain_id: Union[str, int, None] = None):
    """Convert Biopython Model to a dict {'chain_id': aa_sequence, }

    Parameters
    ----------
    model: Biopython Model
    chain_id: select chain_id to convert; if None, return all chains

    Returns
    -------
    ret: a dict with key (chain_id), value (chain amino acid sequence) pair

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bp_read_pdb, bp_model2seq
    >>> structure = bp_read_pdb(pdb_5ni1a_path())
    >>> bp_model2seq(structure[0])
    {'A': 'VLSPADKTNVKAAWGKVGAHAGE...'}

    """
    ret = {}
    if chain_id is not None:
        assert model.has_id(chain_id), f"Chain ID {chain_id} not exists."
        ret[chain_id] = bp_chain2seq(model[chain_id])
    else:
        for chain in model:
            ret[chain.get_id()] = bp_chain2seq(chain)
    return ret


def bp_struc2seq(structure: Structure, model_id: Union[str, int, None] = None, chain_id: Union[str, int, None] = None):
    """Convert Biopython Structure to a nested dict {'model_id': {'chain_id': aa_sequence, }, }

    Parameters
    ----------
    structure: Biopython Structure
    model_id: select model_id; if None, return all models
    chain_id: select chain_id; if None, return all chains

    Returns
    -------
    ret: a nested dict, 1st level key is model_id, 2nd level key is chain_id, value is amino acid sequence

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bp_read_pdb, bp_struc2seq
    >>> structure = bp_read_pdb(pdb_5ni1a_path())
    >>> bp_struc2seq(structure)
    {0: {'A': 'VLSPADKTNVKAAWGKVGAHAGE...'}}

    """
    ret = {}
    if model_id is not None:
        assert structure.has_id(model_id), f"Model ID {model_id} not exists."
        ret[model_id] = bp_model2seq(structure[model_id], chain_id=chain_id)
    else:
        for model in structure:
            ret[model.get_id()] = bp_model2seq(model, chain_id=chain_id)
    return ret


def pdb2seq(file_path: Union[str, Path], model_id=None, chain_id=None):
    """A convenient function to get pdb file sequence dict

    Parameters
    ----------
    file_path: pdb file path
    model_id: select model_id; if None, return all models
    chain_id: select chain_id; if None, return all chains

    Returns
    -------
    ret: a nested dict, 1st level key is model_id, 2nd level key is chain_id, value is amino acid sequence

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bp_read_pdb, pdb2seq
    >>> pdb2seq(pdb_5ni1a_path())
    {0: {'A': 'VLSPADKTNVKAAWGKVGAHAGE...'}}

    """
    s = bp_read_pdb(file_path)
    return bp_struc2seq(s, model_id=model_id, chain_id=chain_id)


def print_pdb_seqs(file_path: Union[str, Path]):
    """A convenient function to print all sequences from the pdb file

    Parameters
    ----------
    file_path: pdb file path

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import print_pdb_seqs
    >>> print_pdb_seqs(pdb_5ni1a_path())
    {0: {'A': 'VLSPADKTNVKAAWGKVGAHAGE...'}}

    """
    pprint(pdb2seq(file_path))


def bt_chain2seq(atom_arr: AtomArray):
    """Convert biotite AtomArray to amino acid sequence

    Parameters
    ----------
    atom_arr: biotite AtomArray (only one chain)

    Returns
    -------
    AA type sequence string

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bt_read_pdb, bt_chain2seq
    >>> atom_arr = bt_read_pdb(pdb_5ni1a_path())[0]
    >>> bt_chain2seq(atom_arr[atom_arr.chain_id == 'A'])
    'VLSPADKTNVKAAWGKVGAHAGE...'

    """
    _, names = bt_struc.get_residues(atom_arr)
    return "".join([bt_seq.ProteinSequence.convert_letter_3to1(n) for n in names])


def bt_model2seq(atom_arr: AtomArray, chain_id=None):
    """Convert biotite AtomArray to a dict {'chain_id': aa_sequence, }

    Parameters
    ----------
    atom_arr: biotite AtomArray ()
    chain_id: select chain_id; if None, return all chains

    Returns
    -------
    ret: a dict with key (chain_id), value (chain amino acid sequence) pair

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bt_read_pdb, bt_model2seq
    >>> atom_arr = bt_read_pdb(pdb_5ni1a_path())[0]
    >>> bt_model2seq(atom_arr)
    {'A': 'VLSPADKTNVKAAWGKVGAHAGE...'}

    """
    ret = {}
    if chain_id is not None:
        assert chain_id in bt_struc.get_chains(atom_arr), f"Chain ID {chain_id} not exists."
        ret[chain_id] = bt_chain2seq(atom_arr[atom_arr.chain_id == chain_id])
    else:
        for chain in bt_struc.chain_iter(atom_arr):
            ret[chain.chain_id[0]] = bt_chain2seq(chain)
    return ret


# --------
# SSE: Secondary Structure Elements
# Constants for biotite DSSP analysis:
# Dictionary to convert 'secStructList' codes to DSSP values
# https://github.com/rcsb/mmtf/blob/master/spec.md#secstructlist
sec_struct_codes = {0: "I", 1: "S", 2: "H", 3: "E", 4: "G", 5: "B", 6: "T", 7: "C"}
# Converter for the DSSP secondary structure elements
# to the classical ones
dssp_to_abc = {"I": "c", "S": "c", "H": "a", "E": "b", "G": "c", "B": "b", "T": "c", "C": "c"}


def bt_arr2sse(atom_arr: bt_struc.AtomArray, bin_path="mkdssp"):
    """Convert biotite AtomArray to SSE

    Parameters
    ----------
    atom_arr: biotite AtomArray (only one chain)
    bin_path: DSSP binary executable path

    Returns
    -------
    sse: an array containing DSSP secondary structure symbols

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bt_read_pdb, bt_arr2sse
    >>> atom_arr = bt_read_pdb(pdb_5ni1a_path())[0]
    >>> bt_arr2sse(atom_arr[atom_arr.chain_id == 'A'])
    array(['C', 'C', 'C', 'H', 'H', ...], dtype='<U1')
    """
    sse = bt_dssp.DsspApp.annotate_sse(atom_arr, bin_path=bin_path)
    return sse


def bt_arr2abc(atom_arr: AtomArray, chain_id=None):
    """Convert biotite AtomArray to biotite style SSE annotation (a: alpha helix, b: beta sheet, c: others)

    Parameters
    ----------
    atom_arr: biotite AtomArray
    chain_id: select chain_id; if None, return all chains

    Returns
    -------
    An array containing the secondary structure elements

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bt_read_pdb, bt_arr2abc
    >>> atom_arr = bt_read_pdb(pdb_5ni1a_path())[0]
    >>> bt_arr2abc(atom_arr)
    array(['c', 'c', 'c', 'a', 'a', ...], dtype='<U1')

    """
    return bt_struc.annotate_sse(atom_arr, chain_id=chain_id)


def bp_pdb2dssp(file_path, model_id=0):
    """Run DSSP by Biopython and parse secondary structure, can only handle one model

    Parameters
    ----------
    file_path: pdb file path
    model_id: select model id

    Returns
    -------
    dssp: parsing results is a dict with key (chain_id, res_id) value
          (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
           NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
           NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bp_pdb2dssp
    >>> dssp = bp_pdb2dssp(pdb_5ni1a_path())
    >>> keys = list(dssp.keys())
    >>> print(keys[3], dssp[keys[3]])
    ('A', (' ', 4, ' ')) (4, 'P', 'H', 0.7720588235294118, -64.4, -27.0, 0, 0.0, 4, -1.4, 0, 0.0, -1, -0.1)
    """
    structure = bp_read_pdb(file_path)
    model = structure[model_id]
    #
    dssp = DSSP(model, file_path)
    return dssp


def bp_dssp2sse(dssp: DSSP):
    """Convert Biopython DSSP object to SSE sequence

    Parameters
    ----------
    dssp: Biopython DSSP object

    Returns
    -------
    ret: a dict with key (chain_id), value (SSE character list) pair

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bp_pdb2dssp, bp_dssp2sse
    >>> dssp = bp_pdb2dssp(pdb_5ni1a_path())
    >>> bp_dssp2sse(dssp)
    {'A': ['-', '-', '-', 'H', 'H', ...], }

    """
    ret = {}
    for k, g in groupby(dssp.keys(), lambda x: x[0]):
        ret[k] = [dssp[cur_k][2] for cur_k in g]
    return ret


def group_chars(chars: Iterable):
    """Group neighbor chars to [('char', indices list), ]

    Parameters
    ----------
    chars: char list or string like: ['c', 'c', 'c', 'a', 'a', ...] or "cccaa"

    Returns
    -------
    A generator containing tuples of ('char', indices list)

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> from cryostar.utils.pdb_tools import bt_read_pdb, bt_arr2abc
    >>> atom_arr = bt_read_pdb(pdb_5ni1a_path())[0]
    >>> abc_list = bt_arr2abc(atom_arr)
    >>> list(group_chars(abc_list))
    [('c', [0, 1, 2]),
     ('a', [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), ...]

    """
    groups = []
    keys = []

    for k, g in groupby(enumerate(chars), lambda x: x[1]):
        groups.append([item[0] for item in g])
        keys.append(k)
    return zip(keys, groups)


# for visualization
# Create 'FeaturePlotter' subclasses
# for drawing the secondary structure features
def sse_to_abc(sse):
    """Convert biotite DsspApp DSSP to abc annotation (a: alpha helix, b: beta sheet, c: others)

    Parameters
    ----------
    sse: an array containing DSSP secondary structure symbols

    Returns
    -------
    an array containing a (alpha helix), b (beta sheet), c (others) symbols

    """
    return np.array([dssp_to_abc[e] for e in sse], dtype="U1")


class HelixPlotter(graphics.FeaturePlotter):

    # Check whether this class is applicable for drawing a feature
    def matches(self, feature):
        if feature.key == "SecStr":
            if "sec_str_type" in feature.qual:
                if feature.qual["sec_str_type"] == "helix":
                    return True
        return False

    # The drawing function itself
    def draw(self, axes, feature, bbox, loc, style_param):
        # Approx. 1 turn per 3.6 residues to resemble natural helix
        n_turns = np.ceil((loc.last - loc.first + 1) / 3.6)
        x_val = np.linspace(0, n_turns * 2 * np.pi, 100)
        # Curve ranges from 0.3 to 0.7
        y_val = (-0.4 * np.sin(x_val) + 1) / 2

        # Transform values for correct location in feature map
        x_val *= bbox.width / (n_turns * 2 * np.pi)
        x_val += bbox.x0
        y_val *= bbox.height
        y_val += bbox.y0

        # Draw white background to overlay the guiding line
        background = Rectangle(bbox.p0, bbox.width, bbox.height, color="white", linewidth=0)
        axes.add_patch(background)
        axes.plot(x_val, y_val, linewidth=2, color=biotite.colors["dimgreen"])


class SheetPlotter(graphics.FeaturePlotter):

    def __init__(self, head_width=0.8, tail_width=0.5):
        super().__init__()
        self._head_width = head_width
        self._tail_width = tail_width

    def matches(self, feature):
        if feature.key == "SecStr":
            if "sec_str_type" in feature.qual:
                if feature.qual["sec_str_type"] == "sheet":
                    return True
        return False

    def draw(self, axes, feature, bbox, loc, style_param):
        x = bbox.x0
        y = bbox.y0 + bbox.height / 2
        dx = bbox.width
        dy = 0

        if loc.defect & bt_seq.Location.Defect.MISS_RIGHT:
            # If the feature extends into the prevoius or next line
            # do not draw an arrow head
            draw_head = False
        else:
            draw_head = True

        axes.add_patch(
            biotite.AdaptiveFancyArrow(
                x,
                y,
                dx,
                dy,
                self._tail_width * bbox.height,
                self._head_width * bbox.height,
                # Create head with 90 degrees tip
                # -> head width/length ratio = 1/2
                head_ratio=0.5,
                draw_head=draw_head,
                color=biotite.colors["orange"],
                linewidth=0))

from Bio.PDB import PDBParser, DSSP, is_aa
import numpy as np

SS8_TO_SS3 = {
    'G': 'H', 'H': 'H', 'I': 'H',
    'E': 'E', 'B': 'E',
    'S': 'C', 'T': 'C', '-': 'C', 'C': 'C'
}

def extract_sec_ids_merge_small_blocks(pdb_file, dssp_exec='mkdssp', min_block_len=3):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    model = structure[0]

    dssp = DSSP(model, pdb_file, dssp=dssp_exec)

    sec_ids_list = []
    residue_chain_ids = []
    block_id = -1
    nucleic_residue_indices = []

    for chain in model:
        chain_id = chain.id
        residues = list(chain.get_residues())
        is_nucleic = any(res.get_resname().strip() in {'DA', 'DT', 'DC', 'DG', 'A', 'U', 'C', 'G'} for res in residues)

        if is_nucleic:
            for res in residues:
                residue_chain_ids.append(chain_id)
                nucleic_residue_indices.append(len(residue_chain_ids) - 1)
                sec_ids_list.append('NA')
        else:
            ss_labels = []
            res_indices = []

            # Step 1: 解析每个残基的 SS3 类型
            for res in residues:
                res_id = res.get_id()
                dssp_key = (chain_id, res_id)
                if not is_aa(res, standard=True):
                    continue
                if dssp_key in dssp:
                    raw_ss = dssp[dssp_key][2]
                else:
                    raw_ss = 'C'
                ss = SS8_TO_SS3.get(raw_ss, 'C')
                ss_labels.append(ss)
                residue_chain_ids.append(chain_id)
                res_indices.append(len(residue_chain_ids) - 1)

            # Step 2: 分块 + merge 短块
            sec_ids_tmp = [None] * len(ss_labels)
            i = 0
            while i < len(ss_labels):
                current_ss = ss_labels[i]
                j = i
                while j < len(ss_labels) and ss_labels[j] == current_ss:
                    j += 1
                length = j - i

                if length < min_block_len and block_id >= 0:
                    # merge 到前一个 block
                    sec_ids_tmp[i:j] = [block_id] * length
                else:
                    block_id += 1
                    sec_ids_tmp[i:j] = [block_id] * length
                i = j

            # 将编号添加进总列表
            for idx in res_indices:
                sec_ids_list.append(sec_ids_tmp.pop(0))

    # 处理核酸残基：统一编号为下一个 block_id
    if nucleic_residue_indices:
        block_id += 1
        for i in nucleic_residue_indices:
            sec_ids_list[i] = block_id

    sec_ids = np.array(sec_ids_list)
    return sec_ids, np.array(residue_chain_ids)

import torch
from torch_scatter import scatter_mean
from torch_cluster import knn_graph,radius_graph  # pip install torch-cluster

def build_metagraph_radius_from_centroids(pos, sec_ids, k=8):
    """
    pos: (N_residues, 3) - 残基坐标
    sec_ids: (N_residues,) - 每个残基所属 metanode 编号
    k: 每个 metanode 连接几个邻居
    """
    sec_ids = torch.tensor(sec_ids, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float)

    # [1] 聚合质心坐标：每个 metanode 的坐标是其成员残基的均值
    centroids = scatter_mean(pos, sec_ids, dim=0)  # shape: (N_meta, 3)

    # [2] 构建 KNN 图（返回 edge_index: (2, E)）
    edge_index = radius_graph(centroids, r=30 , loop=True)

    return edge_index, centroids

def build_metagraph_knn_from_centroids(pos, sec_ids, k=8):
    """
    pos: (N_residues, 3) - 残基坐标
    sec_ids: (N_residues,) - 每个残基所属 metanode 编号
    k: 每个 metanode 连接几个邻居
    """
    sec_ids = torch.tensor(sec_ids, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float)

    # [1] 聚合质心坐标：每个 metanode 的坐标是其成员残基的均值
    centroids = scatter_mean(pos, sec_ids, dim=0)  # shape: (N_meta, 3)

    # [2] 构建 KNN 图（返回 edge_index: (2, E)）
    edge_index = knn_graph(centroids, k=k, loop=True)

    return edge_index, centroids
# Helper function to convert secondary structure array to annotation
# and visualize it
def visualize_secondary_structure(sse, first_id):
    """Convert biotite style sse ['c', 'a', 'a', ...] to a visualization

    Parameters
    ----------
    sse: biotite annotated sse list
    first_id: first residue id, for showing res_id range accurately

    Returns
    -------
    fig: pyplot figure to be saved

    Examples
    --------
    >>> from cryostar.data import pdb_5ni1a_path
    >>> import biotite.structure as struc
    >>> atom_arr = bt_read_pdb(pdb_5ni1a_path())[0]
    >>> sse = struc.annotate_sse(atom_arr, chain_id="A")
    >>> fig = visualize_secondary_structure(sse, atom_arr.res_id[0])
    >>> fig.savefig("test.png", bbox_inches='tight', dpi=150)

    """

    def _add_sec_str(annotation, first, last, str_type):
        if str_type == "a":
            str_type = "helix"
        elif str_type == "b":
            str_type = "sheet"
        else:
            # coil
            return
        feature = bt_seq.Feature("SecStr", [bt_seq.Location(first, last)], {"sec_str_type": str_type})
        annotation.add_feature(feature)

    # Find the intervals for each secondary structure element
    # and add to annotation
    annotation = bt_seq.Annotation()
    curr_sse = None
    curr_start = None
    for i in range(len(sse)):
        if curr_start is None:
            curr_start = i
            curr_sse = sse[i]
        else:
            if sse[i] != sse[i - 1]:
                _add_sec_str(annotation, curr_start + first_id, i - 1 + first_id, curr_sse)
                curr_start = i
                curr_sse = sse[i]
    # Add last secondary structure element to annotation
    _add_sec_str(annotation, curr_start + first_id, i - 1 + first_id, curr_sse)

    fig = plt.figure(figsize=(8.0, 3.0))
    ax = fig.add_subplot(111)
    graphics.plot_feature_map(ax,
                              annotation,
                              symbols_per_line=150,
                              loc_range=(first_id, first_id + len(sse)),
                              show_numbers=True,
                              show_line_position=True,
                              feature_plotters=[HelixPlotter(), SheetPlotter()])
    fig.tight_layout()
    return fig

def affine_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane):
    rotation = rots_from_two_vecs(
        e1_unnormalized=origin - point_on_neg_x_axis,
        e2_unnormalized=point_on_xy_plane - origin,
    )
    return torch.cat((rotation, origin[...,None]),dim=-1)

def rots_from_two_vecs(e1_unnormalized, e2_unnormalized):
    e1 = torch.nn.functional.normalize(e1_unnormalized, p=2, dim=-1)
    c = torch.einsum("...i,...i->...", e2_unnormalized, e1)[..., None]  # dot product
    e2 = e2_unnormalized - c * e1
    e2 = torch.nn.functional.normalize(e2, p=2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)

def update_bb(quaternion, update_ca, ori_ca, ori_bb):
    y = torch.cat([torch.sqrt(torch.square(torch.tensor(1.5))+torch.tensor(1e-6))*torch.ones(size=quaternion.shape[:-1]+(1,),dtype=torch.float,device=quaternion.device),quaternion],dim=-1)
    rotation = quaternion_to_matrix(y)
    translation = update_ca - ori_ca[None]
    new_bb = ori_bb[None] + translation[:,:,None,:]
    delta_bb = new_bb - update_ca[...,None,:]
    # import pdb;pdb.set_trace()
    new_bb = delta_bb * rotation.transpose(-1,-2) + update_ca[...,None,:]
    return new_bb
    # new_affine = affine_composition(affine,torch.cat((rotation,translation),dim=-1))
    # return new_affine
def affine_composition(a1, a2):
    """
    Does the operation a1 o a2
    """
    rotation = get_affine_rot(a1) @ get_affine_rot(a2)
    translation = affine_mul_vecs(a1, get_affine_translation(a2))
    return torch.cat((rotation, translation[..., None]), dim=-1)

def affine_mul_vecs(affine, vecs):
    num_unsqueeze_dims = len(vecs.shape) - len(affine.shape) + 1
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    return torch.einsum(
        "...ij, ...j-> ...i", get_affine_rot(affine), vecs
    ) + get_affine_translation(affine)
def get_affine_translation(affine):
    return affine[..., :, -1]
def get_affine_rot(affine):
    return affine[..., :3, :3]
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def replace_and_save_pdb(reference_pdb, new_coords, output_pdb):
    # 加载参考PDB文件
    structure = bt_struc.io.load_structure(reference_pdb)
    backbone = []
    for atom in structure:
        if atom.atom_name in ['N','CA','C']:
            backbone.append(atom)
    # 初始化一个空的 AtomArrayStack，用于存储多个模型
    atom_stack = bt_struc.AtomArrayStack(depth=new_coords.shape[0], length=new_coords.shape[1])
    new_coords = new_coords.cpu().numpy()
    # 假设 new_coords 是一个形状为 (N, 3 * N_res, 3) 的数组
    # 其中 N 是模型数，N_res 是残基数，3 * N_res 是每个模型中 N、CA、C 原子的总数
    for model_idx in range(new_coords.shape[0]):
        # 获取参考结构中的模型
        model = np.array(backbone)
        # new_atoms = []
        for res_idx in range(new_coords.shape[1]//3):
            # 获取当前残基的 N、CA、C 原子
            n_atom = model[3 * res_idx]       # N 原子
            ca_atom = model[3 * res_idx + 1]  # CA 原子
            c_atom = model[3 * res_idx + 2]   # C 原子
            
            # 从 new_coords 中获取新坐标
            new_n_coords = new_coords[model_idx, res_idx * 3]  # N 原子坐标
            new_ca_coords = new_coords[model_idx, res_idx * 3 + 1]# CA 原子坐标
            new_c_coords = new_coords[model_idx, res_idx * 3 + 2]  # C 原子坐标
            
            # 更新坐标
            n_atom.coord = new_n_coords
            ca_atom.coord = new_ca_coords
            c_atom.coord = new_c_coords
            
            # 将这些原子添加到 new_atoms 列表中
            # new_atoms.append(n_atom)
            # new_atoms.append(ca_atom)
            # new_atoms.append(c_atom)
        
            atom_stack[model_idx][res_idx *3 ] = n_atom
            atom_stack[model_idx][res_idx *3 +1] = ca_atom
            atom_stack[model_idx][res_idx *3 +2] = c_atom
        # new_atom_array = bt_struc.AtomArray(np.array(new_atoms))
        # atom_stack.append(new_atom_array)
    bt_struc.io.save_structure(output_pdb,atom_stack)
    
def get_kplus_neighbor_metanodes(node_coords, segment_ids, metanode_coords):
    dist = torch.cdist(node_coords, metanode_coords, p=2)  # [N, S]
    N, S = dist.size()
    device = node_coords.device

    # Exclude self
    dist_masked = dist.clone()
    dist_masked[torch.arange(N), segment_ids] = float('inf')
    top4_dist, top4_idx = torch.topk(dist_masked, 4, dim=1, largest=False)

    edge_index = []
    edge_vector = []

    for i in range(N):
        sid = segment_ids[i].item()
        top_idx = top4_idx[i].tolist()

        neighbor_idx = []
        if sid > 0:
            neighbor_idx.append(sid - 1)
        if sid < S - 1:
            neighbor_idx.append(sid + 1)

        combined = top_idx + neighbor_idx
        unique_idx = list(dict.fromkeys(combined))

        for j in unique_idx:
            edge_index.append([j, i])  # metanode → node
            edge_vector.append((node_coords[i] - metanode_coords[j]).tolist())  # direction: meta → node

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).T  # [2, E]
    edge_vector = torch.tensor(edge_vector, dtype=torch.float, device=device)  # [E, 3]

    return edge_index, edge_vector
