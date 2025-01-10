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

# from esm.models.esm3 import ESM3

# model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

# with torch.autocast(enabled=True, device_type=torch.device("cuda").type, dtype=torch.bfloat16): 
#     xx_ids = torch.tensor([[1,2,3,4,5]], device="cuda")
#     xx = model(structure_tokens=xx_ids)
#     xx.shape

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

    with open("/cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed/af_swissprot_str.pkl", 'rb') as f:
        struct_data = pickle.load(f)
        
    struct_seq = list(struct_data.keys())

    seq = struct_seq[2]
    target_name = "O95405_894"

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
        coordinates, sequence=seq)
    
    chain.to_pdb(f"/home/liuzijing/workspace/{target_name}.pdb")



        

        



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