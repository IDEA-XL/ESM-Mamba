import torch
import numpy as np
from esm.tokenization import EsmSequenceTokenizer

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
    pad_token_id = EsmSequenceTokenizer().pad_token_id
    
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
    pad_token_id = EsmSequenceTokenizer().pad_token_id
    
    seqs, masks = pad_sequences(seqs, pad_token_id, return_mask=True)
    inputs["input_ids"] = seqs
    inputs["labels"] = label_ids
    inputs["attention_mask"] = masks
    inputs["structure_seq_mask"] = structure_seq_masks
    return inputs

def collate_fn_slm(batch):
    inputs = {}
    seqs, label_ids, structure_seq_masks = tuple(zip(*batch))

    breakpoint()

    label_ids = pad_sequences(label_ids, -100)
    structure_seq_masks = pad_sequences(structure_seq_masks, False)
    pad_token_id = EsmSequenceTokenizer().pad_token_id
    
    seqs, masks = pad_sequences(seqs, pad_token_id, return_mask=True)
    inputs["input_ids"] = seqs
    inputs["labels"] = label_ids
    inputs["attention_mask"] = masks
    inputs["structure_seq_mask"] = structure_seq_masks
    return inputs