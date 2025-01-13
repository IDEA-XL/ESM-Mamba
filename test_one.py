import torch
import numpy as np

import glob
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

@torch.inference_mode()
def test_one(input_seq, model, ckpt_path,
             target_name: str = "",
             temperature: float = 1,
             parallel_size: int = 4,
             do_sample: bool = True):
    
    structure_tokenizer = StructureTokenizer()

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

        breakpoint() #  probs[0].sort().values[-5:],probs[0].sort().indices[-5:]

        ## multinomial
        # next_token = torch.multinomial(probs, num_samples=1)
        if do_sample:

        ## top k
            top_k_probs, top_k_indices = torch.topk(logits, 5, axis=-1)
            probs = torch.softmax(top_k_probs, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            for k in range(parallel_size):
                next_token[k] = top_k_indices[k, next_token[k]]

        else:
             ## greedy
            next_token = torch.argmax(probs, dim=-1)[:, None]

        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        inputs_embeds = model.prepare_gen_img_embeds(next_token)

    breakpoint()
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

        chain.to_pdb(f"{ckpt_path}/test_{target_name}{i}.pdb")


if __name__ == "__main__":
    ckpt = sys.argv[1]
    my_device = "cuda"

    # >Q92FR5
    input_seq = "MMNRVVLVGRLTKDPELRYTPAGVAVATFTLAVNRTFTNQQGEREADFINCVVWRKPAENVANFLKKGSMAGVDGRVQTRNYEGNDGKRVYVTEIVAESVQFLE"
    target_name = "Q92FR5"

    # O95405_1331
    # input_seq = "HSRLTEHVAKAFCLALCPHLKLLKEDGMTKLGLRVTLDSDQVGYQAGSNGQPLPSQYMNDLDSALVPVIHGGACQLSEGPVVMELIFYILEN"
    # target_name = "O95405_1331"

    #
    # input_seq = "AMNLIPEDGLPPILISTGVKGDYAVEEKPSQISVMQQLEDGGPDPLVFVLNANLLSMVKIVNYVNRKCWCFTTKGMHAVGQSEIVILLQCLPDEKCLPKDIFNHFVQLYRDALAGNVVSNLGHSFFSQSFLGSKEHGGFLYVTSTYQSLQDLVLPTPPYLFGILIQKWETPWAKVFPIRLMLRLGAEYRLYPCPLFSVRFRKPLFGETGHTIMNLLADFRNYQYTLPVVQGLVVDMEVRKTSIKIPSNRYNEMMKAMNKSNEHVLAGGACFNEKADSHLVCVQNDDGNYQTQAISIHNQPRKVTGASFFVFSGALKSSSGYLAKSSIVEDGVMVQITAENMDSLRQALREMKDFTITCGKADAEEPQEHIHIQWVDDDKNVSKGVVSPIDGKSMETITNVKIFHGSEYKANGKVIRWTEVFF"
    # target_name = "O95405_894"

    # 2hz4
    # input_seq = "VSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQES"

    # input_seq = "VSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECN"

    model = MultiModalityCausalLM.from_pretrained(ckpt, device_map=my_device, gradient_checkpointing=False, use_cache = True)
    model = model.to(torch.bfloat16).cuda().eval()

    # cameo_dir = "/cto_studio/xtalpi_lab/liuzijing/temp/modeling/2024.10.05"
    # f_list = glob.glob(cameo_dir + '/*')

    # for f1 in f_list:
    #     target_name = f1.split("/")[-1]
    #     fasta_file = f1 + "/target.fasta"
    #     print(target_name)
    #     with open(fasta_file, 'r') as f:
    #         txt = f.read()
    #         input_seq = txt.split('\n')[-1]
    #     print(len(input_seq))
    #     if len(input_seq) > 510:
    #         continue
    
    test_one(input_seq, model, ckpt, target_name)
