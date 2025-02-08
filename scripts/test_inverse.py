# Test generation of structure given sequences
import torch
import os
import sys
import glob

from tokenizers import Tokenizer

from janus_prot.model.modeling_ss import MultiModalityCausalLM
from esm.models.vqvae import StructureTokenDecoder
from esm.tokenization import StructureTokenizer
from esm.utils import decoding
from esm.utils.structure.protein_chain import ProteinChain

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig


ESM3_CKPT_ROOT = "/cto_studio/xtalpi_lab/liuzijing/weights/esm3-sm-open-v1/data/weights"

def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        ESM3_CKPT_ROOT+"/esm3_structure_decoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model

esm3_model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

@torch.inference_mode()
def test_one(input_pdb, model,
             target_name: str = "",
             temperature: float = 1,
             parallel_size: int = 4,
             do_sample: bool = True):

    protein_chain = ProteinChain.from_pdb(input_pdb)
    protein = ESMProtein.from_protein_chain(protein_chain)
    pdb_seq = protein.sequence

    protein.sequence = None
    with torch.autocast(enabled=True, device_type=torch.device("cuda").type, dtype=torch.bfloat16):
        protein_tensor = esm3_model.encode(protein)

        output = esm3_model.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
        struct_embedding = output.per_residue_embedding[1:-1,:].detach()

    len_seq = struct_embedding.shape[0]

    seq = "3" + "G"*len_seq + "4" + "1"

    with open("janus_prot/model/progen/tokenizer.json", 'r') as f:
        progen_tokenizer = Tokenizer.from_str(f.read())

    input_ids = torch.tensor(progen_tokenizer.encode(seq).ids, dtype=torch.long)

    tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size):
        tokens[i, :] = input_ids
    inputs_embeds = model.language_model.transformer.wte(tokens)
    
    inputs_embeds[:,1:len_seq+1] = model.aligner(struct_embedding.to(torch.bfloat16))

    outputs = model.language_model.generate(inputs_embeds=inputs_embeds,
                max_new_tokens=510,
                do_sample=False,
                use_cache=True,)
    
    output_seq = progen_tokenizer.decode(outputs[0].cpu().tolist())
    return output_seq[0:len_seq]


if __name__ == "__main__":
    ckpt = sys.argv[1]
    my_device = "cuda"

    model = MultiModalityCausalLM.from_pretrained(ckpt, device_map=my_device, gradient_checkpointing=False, use_cache = True)
    model = model.to(torch.bfloat16).cuda().eval()

    test_pdb = "/home/liuzijing/workspace/Q92FR5_0_0.pdb"
    target_name = "Q92FR5"

    test_seq = test_one(test_pdb, model, target_name=target_name)
    print(test_seq)
