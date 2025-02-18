import torch
import numpy as np
# from esm.tokenization import EsmSequenceTokenizer

progen_pad_token_id = 0

def pad_sequences(sequences, constant_value=0, dtype=None, return_mask=False) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
        attention_mask = np.full(shape, 0, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        device = sequences[0].device
        array = torch.full(shape, constant_value, dtype=dtype, device=device)
        attention_mask = torch.full(shape, 0, dtype=dtype, device=device)

    for mask, arr, seq in zip(attention_mask, array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq
        mask[arrslice] = 1

    if return_mask:
        return array, attention_mask
    else:
        return array


def collate_fn_gpt(batch):
    inputs = {}
    seqs, label_ids = tuple(zip(*batch))

    label_ids = pad_sequences(label_ids, -100)
    pad_token_id = progen_pad_token_id
    
    seqs, masks = pad_sequences(seqs, pad_token_id, return_mask=True)
    inputs["input_ids"] = seqs
    inputs["labels"] = label_ids
    inputs["attention_mask"] = masks
    return inputs
    

def collate_fn_mm(batch):
    inputs = {}
    seqs, label_ids, structure_seq_masks = tuple(zip(*batch))

    label_ids = pad_sequences(label_ids, -100)
    structure_seq_masks = pad_sequences(structure_seq_masks, False)
    pad_token_id = progen_pad_token_id
    
    seqs, masks = pad_sequences(seqs, pad_token_id, return_mask=True)
    inputs["input_ids"] = seqs
    inputs["labels"] = label_ids
    inputs["attention_mask"] = masks
    inputs["structure_seq_mask"] = structure_seq_masks
    return inputs

def collate_fn_slm(batch):
    inputs = {}
    seqs, label_ids, structure_seq_masks = tuple(zip(*batch))
    label_ids = pad_sequences(label_ids, -100)

    batch_size = len(batch)
    struct2seq_id = torch.full([batch_size], False, dtype=torch.bool, device=label_ids[0].device)
    struct_emb_mask = torch.full(label_ids.shape, False, dtype=torch.bool, device=label_ids.device)
    struct_emb = None
    for i in range(batch_size):
        if len(batch[i]) == 3:
            struct2seq_id[i] = False
        elif len(batch[i]) == 5:
            struct2seq_id[i] = True
            struct_emb_mask[i, 0:batch[i][3].shape[0]] = batch[i][3]
            if struct_emb is None:
                struct_emb = batch[i][4]
            else:
                struct_emb = torch.cat((struct_emb, batch[i][4]))
        elif len(batch[i]) == 4:
            struct2seq_id[i] = True

    inputs["struct_emb"] = struct_emb
    inputs["struct_emb_mask"] = struct_emb_mask
    inputs["struct2seq_id"] = struct2seq_id

    structure_seq_masks = pad_sequences(structure_seq_masks, False)
    pad_token_id = progen_pad_token_id
    
    seqs, masks = pad_sequences(seqs, pad_token_id, return_mask=True)
    inputs["input_ids"] = seqs
    inputs["labels"] = label_ids
    inputs["attention_mask"] = masks
    inputs["structure_seq_mask"] = structure_seq_masks
    return inputs