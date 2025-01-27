import sys
import torch
import pandas as pd
from transformers import LlamaForCausalLM

from esm.models.vqvae import StructureTokenDecoder
from esm.utils import encoding, decoding
from esm.tokenization import EsmSequenceTokenizer, StructureTokenizer
from esm.utils.structure.protein_chain import ProteinChain
from janus_prot.data.constants import SEQ_OFFSET

my_device = "cuda"
ESM3_CKPT_ROOT = "/cto_studio/xtalpi_lab/liuzijing/weights/esm3-sm-open-v1/data/weights"

def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        ESM3_CKPT_ROOT+"/esm3_structure_decoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model


def test_one(input_seq, model, decoder, output_path):

    sequence_tokenizer = EsmSequenceTokenizer()
    structure_tokenizer = StructureTokenizer()
    token_ids = encoding.tokenize_sequence(input_seq, sequence_tokenizer, add_special_tokens=True)

    token_ids = token_ids[None, :].to(my_device) 

    generate_ids = model.generate(
        token_ids, 
        max_length=1024, 
        do_sample=True, 
        top_k=0, 
        top_p=0.95, 
        temperature=0.8, 
        num_return_sequences=5
    )

    seq_len = len(input_seq)
    struct_token_1 = generate_ids[:, seq_len+2:2*seq_len+4] - SEQ_OFFSET
    struct_token = generate_ids[:, seq_len+3:2*seq_len+3] - SEQ_OFFSET
    struct_token = torch.cat((struct_token[:,0:1], struct_token,struct_token[:,-1:]), dim=1)
    struct_token[:,0] = structure_tokenizer.bos_token_id
    struct_token[:,-1] = structure_tokenizer.eos_token_id

    assert torch.all(torch.isclose(struct_token[:,0], struct_token_1[:,0]))
    if not torch.all(torch.isclose(struct_token[:,-1], struct_token_1[:,-1])):
        print("eos not correctly predicted, cut with seq length")
    if (struct_token < 0).sum() > 0:
        print("negative strucutre token exists, replace with pad token")
        struct_token[struct_token<0] = structure_tokenizer.pad_token_id

    for i in range(struct_token.shape[0]):
        coordinates, plddt, ptm = decoding.decode_structure(
                structure_tokens=struct_token[i],
                structure_decoder=decoder,
                structure_tokenizer=structure_tokenizer,
                sequence=input_seq,
            )
        
        chain = ProteinChain.from_atom37(
            coordinates, sequence=input_seq)
        
        chain.to_pdb(f"{output_path}/test{i}.pdb")


if __name__ == "__main__":
    ckpt = sys.argv[1]
    model = LlamaForCausalLM.from_pretrained(ckpt, device_map=my_device)
    decoder = ESM3_structure_decoder_v0(my_device)

    data_df = pd.read_csv("/cto_studio/xtalpi_lab/liuzijing/temp/cameo.csv")

    for row in data_df.itertuples(index=False):
        pdb_id = row.pdb_id
        test_input = row.seq
        gt_pdb = f"/cto_studio/xtalpi_lab/liuzijing/temp/cameo_202407_202409/{pdb_id}/target.pdb"
        output_path = f"/cto_studio/xtalpi_lab/liuzijing/temp/cameo_202407_202409/{pdb_id}"
        test_one(test_input, model, decoder, output_path)


    
