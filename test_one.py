import torch
import numpy as np

from tokenizers import Tokenizer

from model.progen.modeling_progen import ProGenForCausalLM
from model.progen.configuration_progen import ProGenConfig

from model.modeling_ss import MultiModalityCausalLM, MultiModalityConfig

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


def test_one(input_seq, ckpt_path, 
             temperature: float = 1,
             parallel_size: int = 4):
    my_device = "cuda"

    structure_tokenizer = StructureTokenizer()

    model = MultiModalityCausalLM.from_pretrained(ckpt_path, device_map=my_device, gradient_checkpointing=False, use_cache = True)
    model = model.to(torch.float16).cuda().eval()

    with open("model/progen/tokenizer.json", 'r') as f:
        progen_tokenizer = Tokenizer.from_str(f.read())

    len_seq = len(input_seq)

    seq = "1" + input_seq + "2" + "3"

    input_ids = torch.tensor(progen_tokenizer.encode(seq).ids, dtype=torch.long)

    tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.int).cuda()

    for i in range(parallel_size):
        tokens[i, :] = input_ids

    inputs_embeds = model.language_model.transformer.wte(tokens)

    generated_tokens = torch.zeros((parallel_size, len_seq), dtype=torch.int).cuda()

    for i in range(len_seq):
        outputs = model.language_model.transformer(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state #(B, L, d)

        logits = model.gen_head(hidden_states[:, -1, :]).to(torch.float32) # last logit

        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        inputs_embeds = model.prepare_gen_img_embeds(next_token)

    struct_token = generated_tokens
    struct_token = torch.cat((struct_token[:,0:1], struct_token,struct_token[:,-1:]), dim=1)
    struct_token[:,0] = structure_tokenizer.bos_token_id
    struct_token[:,-1] = structure_tokenizer.eos_token_id

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
    ckpt = sys.argv[1]

    test_input = "MMNRVVLVGRLTKDPELRYTPAGVAVATFTLAVNRTFTNQQGEREADFINCVVWRKPAENVANFLKKGSMAGVDGRVQTRNYEGNDGKRVYVTEIVAESVQFLE"
    # ckpt = "/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/checkpoint-10000"
    test_one(test_input, ckpt)
