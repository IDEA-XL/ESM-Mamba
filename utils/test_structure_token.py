# test ESM3 structure token and ESM3 model
from typing import Callable

import torch
import gzip
import glob
import numpy as np
# import torch.nn as nn
from tqdm import tqdm
import pickle
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

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

f_name = "/cto_studio/xtalpi_lab/Datasets/af_swissprot/AF-Q92FR5-F1-model_v4.pdb.gz"
with gzip.open(f_name, 'rt', encoding='utf-8') as f:
    protein = ESMProtein.from_pdb(f, is_predicted=False)
protein.sequence = None
# protein = ESMProtein.from_pdb("/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/AF-Q92FR5-F1-model_v4.pdb", is_predicted=False)
breakpoint()
with torch.autocast(enabled=True, device_type=torch.device("cuda").type, dtype=torch.bfloat16):
    protein_tensor = model.encode(protein)

    output = model.forward_and_sample(
        protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
    tosave = output.per_residue_embedding[1:-1,:].detach().cpu().numpy()


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

    sequence_tokenizer = EsmSequenceTokenizer()
    structure_tokenizer = StructureTokenizer()
    structure_encoder = ESM3_structure_encoder_v0(my_device)

    data_dir = '/cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed'
    f_list = glob.glob(data_dir + '/*.pkl')

    fasta_all = '/cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed'

    # for f_name in tqdm(f_list):
    #     pname = f_name.split("/")[-1].split('.')[0]
    #     fasta_name = data_dir + f"/{pname[:-4]}.fasta"
    #     str_name = f_name

    #     with open(fasta_name, 'r') as f:
    #         texts = f.read()

    # qfasta = "MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVL"
    # /cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed/UP000002311.fasta

    # /cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed/UP000002485.fasta

    with open("/cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed/af_swissprot_str.pkl", 'rb') as f:
        struct_data = pickle.load(f)
        
    struct_seq = list(struct_data.keys())

    seq = struct_seq[4]
    target_name = "Q92FR5_0"

    # seq = qfasta
    # target_name = "UP000002485"

    struct_token = torch.tensor(struct_data[seq], dtype=torch.int64, device=my_device)

    struct_token = torch.cat((struct_token[0:1], struct_token, struct_token[-1:]), dim=0)
    struct_token[0] = structure_tokenizer.bos_token_id
    struct_token[-1] = structure_tokenizer.eos_token_id

    coordinates, plddt, ptm = decoding.decode_structure(
            structure_tokens=struct_token,
            structure_decoder=ESM3_structure_decoder_v0(my_device),
            structure_tokenizer=structure_tokenizer,
            sequence=seq,
        )
    
    chain = ProteinChain.from_atom37(
        coordinates, sequence=seq, confidence=plddt)
    
    chain.to_pdb(f"/home/liuzijing/workspace/{target_name}.pdb")


# first res trans [  1.3696, -14.6521,  19.3721] 
# rot tensor([[ 0.2052,  0.8345,  0.5114],
#         [-0.0680, -0.5091,  0.8580],
#         [ 0.9764, -0.2108, -0.0477]],

# t_atoms_to_global[0,0,0:3].trans
# tensor([[  1.3696, -14.6521,  19.3721],
#         [  1.3696, -14.6521,  19.3721],
#         [  1.3696, -14.6521,  19.3721]], device='cuda:0',
#        grad_fn=<SliceBackward0>)
# (Pdb) t_atoms_to_global[0,0,0:3].rot._rots
# tensor([[[ 0.2052,  0.8345,  0.5114],
#          [-0.0680, -0.5091,  0.8580],
#          [ 0.9764, -0.2108, -0.0477]],

#         [[ 0.2052,  0.8345,  0.5114],
#          [-0.0680, -0.5091,  0.8580],
#          [ 0.9764, -0.2108, -0.0477]],

#         [[ 0.2052,  0.8345,  0.5114],
#          [-0.0680, -0.5091,  0.8580],
#          [ 0.9764, -0.2108, -0.0477]]], device='cuda:0',
#        grad_fn=<SliceBackward0>)

# t_atoms_to_global[0,0,2:3].rot.apply(lit_positions[0,0,2:3]) + t_atoms_to_global[0,0,2:3].trans
# rigids[0,1, None].apply(all_bb_coords_local[0,1,2:3])

# lit_positions[0,0,0:3] @ t_atoms_to_global[0,0,0:3].rot._rots.transpose(-1, -2).squeeze(-3) + t_atoms_to_global[0,0,0:3].trans

# # p @ self._rots.transpose(-1, -2).squeeze(-3)

# all_bb_coords_local[0,0] @ rigids[0,1, None].rot._rots.transpose(-1, -2).squeeze(-3) + rigids[0,1, None].trans

    # out_plddt = {}
    # out_str = {}
    # fasta_name = "/cto_studio/xtalpi_lab/Datasets/af_swissprot.fasta"
    # plddt_name = "/cto_studio/xtalpi_lab/Datasets/af_swissprot_plddt.pkl"
    # f0 = open(fasta_name, 'w')

    # for f_name in tqdm(f_list):

    #     uniprot = f_name.split('/')[-1].split("-")[1]

    #     with gzip.open(f_name, 'rt', encoding='utf-8') as f:
    #         protein = ProteinChain.from_pdb(f, is_predicted=True)

    #     _, _, structure_tokens = encoding.tokenize_structure(
    #             torch.tensor(protein.atom37_positions, dtype=torch.float32),
    #             structure_encoder,
    #             structure_tokenizer=structure_tokenizer,
    #             reference_sequence=protein.sequence or "",
    #             add_special_tokens=True,
    #         )
        
    #     f0.write('>' + uniprot + '\n' )
    #     f0.write(protein.sequence + '\n')

    #     out_str[protein.sequence] = structure_tokens.cpu().numpy()
    #     out_plddt[protein.sequence] = protein.confidence

    # with open(plddt_name, 'wb') as f:
    #     pickle.dump(out_plddt, f)

    # f0.close()
    # ### swissprot

    # f_name = "/cto_studio/xtalpi_lab/Datasets/af_swissprot/AF-Q5VSL9-F1-model_v4.pdb.gz"

    

    # protein = ProteinChain.from_rcsb("1KTS", chain_id="B") #7kn4 8qdg

    # choose the subsequences with plddt > 70 and length > 15
    # idx = np.arange(len(protein.sequence))
    # idx = idx[protein.confidence > 70] 

    # current_seq = [idx[0]]
    # seqs = []
    # seqs_all = []
    # for i in range(1, len(idx)):
    #     if idx[i] == idx[i-1] + 1:
    #         current_seq.append(idx[i])
    #     else:
    #         if len(current_seq) > 15:
    #             seqs_all.extend(current_seq)
    #             seqs.append(current_seq)
    #         current_seq = [idx[i]]

    # if len(current_seq) > 15:
    #     seqs_all.extend(current_seq)
    #     seqs.append(current_seq)

    # breakpoint()

    # sequence_tokens = encoding.tokenize_sequence(
    #             protein.sequence, sequence_tokenizer, add_special_tokens=True
    #         )

    # coords, plddt, residue_index = protein.to_structure_encoder_inputs()
    
    # chain = ProteinChain.from_atom37(
    #     coords, sequence=protein.sequence)
    # breakpoint()

    # from coordiates to tokens
    # _, _, structure_tokens = encoding.tokenize_structure(
    #             torch.tensor(protein.atom37_positions, dtype=torch.float32),
    #             ESM3_structure_encoder_v0(my_device),
    #             structure_tokenizer=structure_tokenizer,
    #             reference_sequence=protein.sequence or "",
    #             add_special_tokens=True,
    #         )
    
    # structure_tokens1 = torch.cat((structure_tokens[seqs[0][0]:seqs[0][-1]], 
    #                                structure_tokens[seqs[1][0]:seqs[1][-1]],
    #                                structure_tokens[seqs[2][0]:seqs[2][-1]],
    #                                structure_tokens[seqs[3][0]:seqs[3][-1]],
    #                                structure_tokens[seqs[4][0]:seqs[4][-1]]))
    # structure_tokens0 = structure_tokens[1:-1]
    # structure_tokens1 = torch.cat((structure_tokens[0:1], structure_tokens0[seqs_all],structure_tokens[-1:]))
    # sequence1 = slice_python_object_as_numpy(protein.sequence, seqs_all)
    
    # _, _, structure_tokens2 = encoding.tokenize_structure(
    #             torch.tensor(protein.atom37_positions[seqs_all], dtype=torch.float32),
    #             ESM3_structure_encoder_v0(my_device),
    #             structure_tokenizer=structure_tokenizer,
    #             reference_sequence=sequence1 or "",
    #             add_special_tokens=True,
    #         )


    # breakpoint()
    # from tokens to coordinates
    # coordinates, plddt, ptm = decoding.decode_structure(
    #         structure_tokens=structure_tokens,
    #         structure_decoder=ESM3_structure_decoder_v0(my_device),
    #         structure_tokenizer=structure_tokenizer,
    #         sequence=protein.sequence,
    #     )
    
    # coordinates1, plddt, ptm = decoding.decode_structure(
    #         structure_tokens=structure_tokens1,
    #         structure_decoder=ESM3_structure_decoder_v0(my_device),
    #         structure_tokenizer=structure_tokenizer,
    #         sequence=sequence1,
    #     )
    
    # coordinates2, plddt, ptm = decoding.decode_structure(
    #         structure_tokens=structure_tokens2,
    #         structure_decoder=ESM3_structure_decoder_v0(my_device),
    #         structure_tokenizer=structure_tokenizer,
    #         sequence=sequence1,
    #     )

    # # from coordinates to pdb file
    # chain = ProteinChain.from_atom37(
    #     coordinates, sequence=protein.sequence)
    
    # chain.to_pdb("/home/liuzijing/workspace/7kn4_A_esm.pdb")