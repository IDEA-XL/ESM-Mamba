# Test generation of structure given sequences
import torch
import os
import sys
import glob
import pickle
import numpy as np
from tokenizers import Tokenizer

from janus_prot.model.modeling_ss import MultiModalityCausalLM
from esm.models.vqvae import StructureTokenDecoder
from esm.tokenization import EsmSequenceTokenizer,StructureTokenizer
from esm.utils import decoding
from esm.utils.structure.protein_chain import ProteinChain


ESM3_CKPT_ROOT = "/cto_studio/xtalpi_lab/liuzijing/weights/esm3-sm-open-v1/data/weights"

def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        ESM3_CKPT_ROOT+"/esm3_structure_decoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model


esm_tokenizer = EsmSequenceTokenizer()

@torch.inference_mode()
def test_one(input_seq, model, ckpt_path, residue_index=None,
             target_name: str = "",
             temperature: float = 1,
             parallel_size: int = 4,
             do_sample: bool = True):
    
    structure_tokenizer = StructureTokenizer()
    seq_tokens = esm_tokenizer.encode(input_seq)
    output = {}
    output["target_name"] = target_name
    output["seq_token"] = seq_tokens
    output["struct_token"] = []
    
    with open("janus_prot/model/progen/tokenizer.json", 'r') as f:
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

        # breakpoint() #  probs[0].sort().values[-5:],probs[0].sort().indices[-5:]

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

    struct_token = generated_tokens
    struct_token = torch.cat((struct_token[:,0:1], struct_token,struct_token[:,-1:]), dim=1)
    struct_token[:,0] = structure_tokenizer.bos_token_id
    struct_token[:,-1] = structure_tokenizer.eos_token_id

    for i in range(parallel_size):
        output["struct_token"].append(struct_token[i].cpu().numpy().tolist())

    if not os.path.exists(f"{ckpt_path}/predictions"):
        os.mkdir(f"{ckpt_path}/predictions")

    plddts = []
    for i in range(struct_token.shape[0]):
        coordinates, plddt, ptm = decoding.decode_structure(
                structure_tokens=struct_token[i],
                structure_decoder=ESM3_structure_decoder_v0(my_device),
                structure_tokenizer=structure_tokenizer,
                sequence=input_seq,
            )

        chain = ProteinChain.from_atom37(
            coordinates, sequence=input_seq, 
            residue_index=residue_index, confidence=plddt)
        plddts.append(plddt.mean())

        chain.to_pdb(f"{ckpt_path}/predictions/pre_{target_name}{i}.pdb")
    plddts_idx = np.array(plddts).argmax()

    return output, plddts_idx


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
    input_seq = "AMNLIPEDGLPPILISTGVKGDYAVEEKPSQISVMQQLEDGGPDPLVFVLNANLLSMVKIVNYVNRKCWCFTTKGMHAVGQSEIVILLQCLPDEKCLPKDIFNHFVQLYRDALAGNVVSNLGHSFFSQSFLGSKEHGGFLYVTSTYQSLQDLVLPTPPYLFGILIQKWETPWAKVFPIRLMLRLGAEYRLYPCPLFSVRFRKPLFGETGHTIMNLLADFRNYQYTLPVVQGLVVDMEVRKTSIKIPSNRYNEMMKAMNKSNEHVLAGGACFNEKADSHLVCVQNDDGNYQTQAISIHNQPRKVTGASFFVFSGALKSSSGYLAKSSIVEDGVMVQITAENMDSLRQALREMKDFTITCGKADAEEPQEHIHIQWVDDDKNVSKGVVSPIDGKSMETITNVKIFHGSEYKANGKVIRWTEVFF"
    target_name = "O95405_894"

    # 2hz4
    # input_seq = "VSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQES"
    # target_name = "2hz4"

    input_seq = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"
    target_name = "BPTI"

    model = MultiModalityCausalLM.from_pretrained(ckpt, device_map=my_device, gradient_checkpointing=False, use_cache = True)
    model = model.to(torch.bfloat16).cuda().eval()

    
    parallel_size = 32
    # test_one(input_seq, model, ckpt, 
    #         None, target_name, parallel_size=parallel_size)

    # cameo_dir = "/cto_studio/xtalpi_lab/liuzijing/temp/modeling/2024.10.05"
    cameo_dir = "/cto_studio/xtalpi_lab/Datasets/casp14-casp15-cameo-test-proteins/casp15_structure_gts"
    f_list = glob.glob(cameo_dir + '/*')
    plddts = {}
    avg_plddts = {}
    rmsds = {}
    avg_rmsds = {}
    tmscores = {}
    tmscores_plddt = {}
    avg_tmscores = {}
    parallel_size = 32
    save_output = {}

    for f1 in f_list:
        # target_name = f1.split("/")[-1]
        # fasta_file = f1 + "/target.fasta"
        # target_pdb = f1 + "/target.pdb"
        # with open(fasta_file, 'r') as f:
        #     txt = f.read()
        #     input_seq = txt.split('\n')[-1]
        target_pdb = f1
        target_name = f1.split("/")[-1].split(".")[0]
        target_pdb = f"/{cameo_dir}/{target_name}.pdb"
        protein_chain = ProteinChain.from_pdb(target_pdb)
        input_seq = protein_chain.sequence

        print(target_name)
        print(len(input_seq))
        if len(input_seq) > 510:
            print("skipped")
            continue
        
        save_output[target_name], pre_idx = test_one(input_seq, model, ckpt, 
                                            protein_chain.residue_index, 
                                            target_name, parallel_size=parallel_size)
        lddt = []
        tmscore = []
        rmsd = []
        for i in range(parallel_size):
            pred_pdb = f"{ckpt}/predictions/pre_{target_name}{i}.pdb"
            result = os.popen(f"lddt -c {pred_pdb} {target_pdb}")
            res = result.read()
            for line in res.splitlines():
                if line.startswith("Global LDDT score"):
                    lddt.append(float(line.split(":")[-1].strip()))

            tm_result = os.popen(f"/cto_studio/xtalpi_lab/softwares/TMscore {pred_pdb} {target_pdb}")
            res = tm_result.read()
            for line in res.splitlines():
                if line.startswith("TM-score"):
                    tmscore.append(float(line.split("=")[1].strip().split(" ")[0]))
                if line.startswith("RMSD of"):
                    rmsd.append(float(line.split("=")[-1].strip()))

        tmscores[target_name] = max(tmscore)
        avg_tmscores[target_name] = np.array(tmscore).mean()
        plddts[target_name] = max(lddt)
        avg_plddts[target_name] = np.array(lddt).mean()
        rmsds[target_name] = min(rmsd)
        avg_rmsds[target_name] = np.array(rmsd).mean()
        tmscores_plddt[target_name] = tmscore[pre_idx]
        breakpoint()
    
    with open(f"{ckpt}/predictions/tokens.pkl", 'wb') as f:
        pickle.dump(save_output, f)

    print("dataset: ", cameo_dir, "model:", ckpt)
    print("average plddts", np.array(list(avg_plddts.values())).mean())
    print("plddts", np.array(list(plddts.values())).mean())
    
    print("average tmscores", np.array(list(avg_tmscores.values())).mean())
    print("tmscores", np.array(list(tmscores.values())).mean())
    print("average rmsds", np.array(list(avg_rmsds.values())).mean())
    print("rmsds", np.array(list(rmsds.values())).mean())

    print("tmscores by plddt", np.array(list(tmscores_plddt.values())).mean())