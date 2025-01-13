### Process pdb files into fasta and ESM3 structure tokens

from typing import Callable

import torch
import gzip
import glob
import sys
import numpy as np
# import torch.nn as nn
from tqdm import tqdm
import pickle

from scipy.spatial.distance import pdist, squareform

# from esm3.models.esm3 import ESM3
# from esm3.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import get_model_tokenizers
from esm.tokenization import EsmSequenceTokenizer, StructureTokenizer
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.misc import slice_python_object_as_numpy

from esm.utils import encoding
from esm.utils import decoding

ckpt_root = "/cto_studio/xtalpi_lab/liuzijing/weights/esm3-sm-open-v1/data/weights"

def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        ckpt_root+"/esm3_structure_decoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model


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

    data_dir = "/cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed" #output dir

    pdb_dir = sys.argv[1]
    pname = pdb_dir.split("/")[-1]

    sequence_tokenizer = EsmSequenceTokenizer()
    structure_tokenizer = StructureTokenizer()
    structure_encoder = ESM3_structure_encoder_v0(my_device)

    f_list = glob.glob(pdb_dir + '/AF-*_v4.pdb.gz')
    out_str = {}
    fasta_name = data_dir + f"/{pname}.fasta"
    str_name = data_dir + f"/{pname}_str.pkl"
    f0 = open(fasta_name, 'w')

    for f_name in tqdm(f_list):

        uniprot = f_name.split('/')[-1].split("-")[1]

        with gzip.open(f_name, 'rt', encoding='utf-8') as f:
            protein = ProteinChain.from_pdb(f, is_predicted=True)

        _, _, structure_tokens = encoding.tokenize_structure(
                torch.tensor(protein.atom37_positions, dtype=torch.float32),
                structure_encoder,
                structure_tokenizer=structure_tokenizer,
                reference_sequence=protein.sequence or "",
                add_special_tokens=True,
            )

        struct = structure_tokens.cpu().numpy()

        plddt = protein.confidence
        seq = protein.sequence

        idx = np.arange(len(seq))
        idx = idx[plddt > 70] 
        if len(idx) < 1:
            continue
        else:
            current_seq = [idx[0]]
            seqs = []
            for i in range(1, len(idx)):
                if idx[i] < idx[i-1] + 12: # sub seq gap < 12
                    current_seq.append(idx[i])
                else:
                    if len(current_seq) > 20:
                        seqs.append(current_seq)
                    current_seq = [idx[i]]

            if len(current_seq) > 20:
                seqs.append(current_seq)
            if len(seqs) < 1:
                continue
            else:
                for seq_ in seqs:
                    start_idx = seq_[0]
                    end_idx = seq_[-1]
                    seq_1 = seq[start_idx:end_idx]
                    pos_ca = protein.atom37_positions[start_idx:end_idx, 1, :]
                    contact_map = np.less(squareform(pdist(pos_ca)), 8.0).astype(np.int64)
                    long_range = np.greater(squareform(pdist(np.arange(len(seq_1))[:, None])), 12).astype(np.int64)
                    if (long_range * contact_map).sum() > len(seq_1):
                        f0.write('>' + f"{uniprot}_{start_idx}" + '\n' )
                        f0.write(seq_1 + '\n')
                        out_str[seq_1] = struct[start_idx+1:end_idx+1]

    with open(str_name, 'wb') as f:
        pickle.dump(out_str, f)

    print(pname, "### total number of structures: ", len(out_str))

    f0.close()

