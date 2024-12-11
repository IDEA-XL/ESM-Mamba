import torch
import os
import glob

import numpy as np
from tqdm import tqdm
from typing import Mapping, Tuple
from esm.utils.structure import mmcif_parsing
from esm.utils.constants import residue_constants
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import get_model_tokenizers
from esm.tokenization import EsmSequenceTokenizer, StructureTokenizer
from esm.utils import encoding

class Error(Exception):
  """Base class for exceptions."""


class MultipleChainsError(Error):
  """An error indicating that multiple chains were found for a given ID."""


class CaDistanceError(Error):
  """An error indicating that a CA atom distance exceeds a threshold."""


def _check_residue_distances(all_positions: np.ndarray,
                             all_positions_mask: np.ndarray,
                             max_ca_ca_distance: float):
  """Checks if the distance between unmasked neighbor residues is ok."""
  ca_position = residue_constants.atom_order['CA']
  prev_is_unmasked = False
  prev_calpha = None
  for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
    this_is_unmasked = bool(mask[ca_position])
    if this_is_unmasked:
      this_calpha = coords[ca_position]
      if prev_is_unmasked:
        distance = np.linalg.norm(this_calpha - prev_calpha)
        if distance > max_ca_ca_distance:
          raise CaDistanceError(
              'The distance between residues %d and %d is %f > limit %f.' % (
                  i, i + 1, distance, max_ca_ca_distance))
      prev_calpha = this_calpha
    prev_is_unmasked = this_is_unmasked

def get_atom_positions(
    mmcif_object: mmcif_parsing.MmcifObject,
    auth_chain_id: str,
    max_ca_ca_distance: float) -> Tuple[np.ndarray, np.ndarray]:
  """Gets atom positions and mask from a list of Biopython Residues."""
  num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

  relevant_chains = [c for c in mmcif_object.structure.get_chains()
                     if c.id == auth_chain_id]
  if len(relevant_chains) != 1:
    raise MultipleChainsError(
        f'Expected exactly one chain in structure with id {auth_chain_id}.')
  chain = relevant_chains[0]

  all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
  all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                dtype=np.int64)
  for res_index in range(num_res):
    pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
    mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
    res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
    if not res_at_position.is_missing:
      res = chain[(res_at_position.hetflag,
                   res_at_position.position.residue_number,
                   res_at_position.position.insertion_code)]
      for atom in res.get_atoms():
        atom_name = atom.get_name()
        x, y, z = atom.get_coord()
        if atom_name in residue_constants.atom_order.keys():
          pos[residue_constants.atom_order[atom_name]] = [x, y, z]
          mask[residue_constants.atom_order[atom_name]] = 1.0
        elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
          # Put the coordinates of the selenium atom in the sulphur column.
          pos[residue_constants.atom_order['SD']] = [x, y, z]
          mask[residue_constants.atom_order['SD']] = 1.0

      # Fix naming errors in arginine residues where NH2 is incorrectly
      # assigned to be closer to CD than NH1.
      cd = residue_constants.atom_order['CD']
      nh1 = residue_constants.atom_order['NH1']
      nh2 = residue_constants.atom_order['NH2']
      if (res.get_resname() == 'ARG' and
          all(mask[atom_index] for atom_index in (cd, nh1, nh2)) and
          (np.linalg.norm(pos[nh1] - pos[cd]) >
           np.linalg.norm(pos[nh2] - pos[cd]))):
        pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
        mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()

    all_positions[res_index] = pos
    all_positions_mask[res_index] = mask
  _check_residue_distances(
      all_positions, all_positions_mask, max_ca_ca_distance)
  return all_positions, all_positions_mask


def load_chain(mmcif_obj, chain_id='A'):
    """Load chain info."""
    all_atom_positions, all_atom_mask = get_atom_positions(mmcif_obj, chain_id, max_ca_ca_distance=float('inf'))
    # Directly parses sequence from fasta, should be consistent to 'aatype' in input features (from .fasta or .pkl)
    sequence = mmcif_obj.chain_to_seqres[chain_id]           
    order_map = residue_constants.restype_order_with_x
    aatype_idx = np.array([order_map.get(rn, order_map['X']) for rn in sequence], dtype=np.int32)
    resolution = np.array([mmcif_obj.header['resolution']], dtype=np.float32)
    return {
        'aatype_index':       aatype_idx,           # [NR,]
        'all_atom_positions': all_atom_positions,   # [NR, 37, 3]
        'all_atom_mask':      all_atom_mask,        # [NR, 37]
        'resolution':         resolution,            # [,]
        'sequence': sequence
    }


ckpt_root = "/cto_studio/xtalpi_lab/liuzijing/weights/esm3-sm-open-v1/data/weights"

def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        ckpt_root+"/esm3_structure_encoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    my_device = "cuda"
    structure_tokenizer = StructureTokenizer()
    structure_encoder = ESM3_structure_encoder_v0(my_device)

    data_dir = "/cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed" #output dir

    cif_folder = '/cto_studio/xtalpi_lab/Datasets/alphafold-datasets/alphafold3/pdb_mmcif/mmcif_files'

    out_str = {}
    fasta_name = data_dir + f"/pdb20220928.fasta"
    str_name = data_dir + f"/pdb20220928_str.pkl"
    f0 = open(fasta_name, 'w')

    f_list = glob.glob(cif_folder + '/*.cif')
    for f_name in tqdm(f_list):
      # f_name = '/cto_studio/xtalpi_lab/Datasets/alphafold-datasets/alphafold3/pdb_mmcif/mmcif_files/1kts.cif'
      pdb_code = f_name.split('/')[-1].split(".")[0]
      with open(f_name, 'r') as f:
          file_data = f.read()

      
      parsing_result = mmcif_parsing.parse(
        file_id=pdb_code, mmcif_string=file_data)
      if parsing_result.mmcif_object is None:
        print(parsing_result.errors)
        continue
      
      chains = list(parsing_result.mmcif_object.chain_to_seqres.keys())

      mmcif_obj_single = parsing_result.mmcif_object

      for chain_id in chains:
        protein = pdb_code + '_' + chain_id
        protein_chain_d = load_chain(mmcif_obj_single, chain_id)
        num_res = protein_chain_d["all_atom_positions"].shape[0]
        atom_positions = np.full(
                [num_res, 37, 3],
                np.nan,
                dtype=np.float32,
            )
        atom_mask = protein_chain_d["all_atom_mask"].astype(bool)
        atom_positions[atom_mask] = protein_chain_d["all_atom_positions"][atom_mask]
        assert num_res == len(parsing_result.mmcif_object.chain_to_seqres[chain_id])

        protein_na = ProteinChain.from_atom37(atom37_positions=atom_positions)
        _, _, structure_tokens = encoding.tokenize_structure(
                    torch.tensor(protein_na.atom37_positions, dtype=torch.float32),
                    structure_encoder,
                    structure_tokenizer=structure_tokenizer,
                    reference_sequence=protein_chain_d['sequence'] or "",
                    add_special_tokens=True,
                )
        s_mask = torch.all(torch.all(torch.isfinite(
            torch.tensor(protein_na.atom37_positions[:,0:3,:], dtype=torch.float32)
            ), dim=-1),dim=-1)
        
        structure_tokens[1:-1][~s_mask] = -100 # 2246 to -100
        
        struct = structure_tokens.cpu().numpy()
        out_str[protein] = struct

        f0.write('>' + f"{protein}" + '\n' )
        f0.write(protein_chain_d['sequence'] + '\n')

f0.close()