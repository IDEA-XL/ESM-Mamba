import torch
import numpy as np

from transformers import LlamaForCausalLM

import sys
sys.path.append('./utils')

from utils.esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from utils.esm.utils import encoding
from utils.esm.tokenization import EsmSequenceTokenizer, StructureTokenizer
from utils.esm.utils import decoding
from utils.esm.utils.structure.protein_chain import ProteinChain

_10TB = 10995116277760
SEQ_OFFSET = 33


ckpt_root = "/cto_studio/xtalpi_lab/liuzijing/weights/esm3-sm-open-v1/data/weights"

def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        ckpt_root+"/esm3_structure_decoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model


def test_one(input_seq, ckpt_path):
    my_device = "cuda"
    sequence_tokenizer = EsmSequenceTokenizer()
    structure_tokenizer = StructureTokenizer()
    token_ids = encoding.tokenize_sequence(input_seq, sequence_tokenizer, add_special_tokens=True)

    model = LlamaForCausalLM.from_pretrained(ckpt_path, device_map=my_device)
    token_ids = token_ids[None, :].to(my_device) 

    generate_ids = model.generate(token_ids, max_length=1024, do_sample=True, top_k=10, top_p=0.95, temperature=0.8, num_return_sequences=5)
    breakpoint()

    seq_len = len(input_seq)
    struct_token_1 = generate_ids[:, seq_len+2:2*seq_len+4] - SEQ_OFFSET
    struct_token = generate_ids[:, seq_len+3:2*seq_len+3] - SEQ_OFFSET
    struct_token = torch.cat((struct_token[:,0:1], struct_token,struct_token[:,-1:]), dim=1)
    struct_token[:,0] = structure_tokenizer.bos_token_id
    struct_token[:,-1] = structure_tokenizer.eos_token_id

    assert torch.all(torch.isclose(struct_token[:,0], struct_token_1[:,0]))
    assert torch.all(torch.isclose(struct_token[:,-1], struct_token_1[:,-1]))
    
    for i in range(struct_token.shape[0]):
        coordinates, plddt, ptm = decoding.decode_structure(
                structure_tokens=struct_token[i],
                structure_decoder=ESM3_structure_decoder_v0(my_device),
                structure_tokenizer=structure_tokenizer,
                sequence=input_seq,
            )
        
        chain = ProteinChain.from_atom37(
            coordinates, sequence=input_seq)
        
        chain.to_pdb(f"{ckpt_path}/test{i}.pdb")


if __name__ == "__main__":

    test_input =  "CPFVVLDNGTHVKPAGCSHLCNGAPETLDNIECYNVTEEVAKRMTPGIPYACWLGWCSKGECKRDNRTEVCYRGS"
    ckpt = "/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/checkpoint-10000"

    test_one(test_input, ckpt)
