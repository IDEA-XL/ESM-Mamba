import warnings

import attr
import torch

from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.tokenization import TokenizerCollectionProtocol
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer,
)
from esm.tokenization.residue_tokenizer import (
    ResidueAnnotationsTokenizer,
)
from esm.tokenization.sasa_tokenizer import (
    SASADiscretizingTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.tokenization.ss_tokenizer import (
    SecondaryStructureTokenizer,
)
from esm.tokenization.structure_tokenizer import (
    StructureTokenizer,
)
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C
from esm.utils.function.encode_decode import (
    decode_function_tokens,
    decode_residue_annotation_tokens,
)
from esm.utils.misc import list_nan_to_none
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

from esm.utils.structure.affine3d import (
    Affine3D,
    RotationMatrix,
)
from esm.utils.constants import residue_constants
from esm.utils.constants.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)

def decode_protein_tensor(
    input: ESMProteinTensor,
    tokenizers: TokenizerCollectionProtocol,
    structure_token_decoder: StructureTokenDecoder,
    function_token_decoder: FunctionTokenDecoder | None = None,
) -> ESMProtein:
    input = attr.evolve(input)  # Make a copy

    sequence = None
    secondary_structure = None
    sasa = None
    function_annotations = []

    coordinates = None

    # If all pad tokens, set to None
    for track in attr.fields(ESMProteinTensor):
        tokens: torch.Tensor | None = getattr(input, track.name)
        if track.name == "coordinates" or track.name == "potential_sequence_of_concern":
            continue
        if tokens is not None:
            tokens = tokens[1:-1]  # Remove BOS and EOS tokens
            tokens = tokens.flatten()  # For multi-track tensors
            track_tokenizer = getattr(tokenizers, track.name)
            if torch.all(tokens == track_tokenizer.pad_token_id):
                setattr(input, track.name, None)
            # If structure track has any mask tokens, do not decode.
            if track.name == "structure" and torch.any(
                tokens == track_tokenizer.mask_token_id
            ):
                setattr(input, track.name, None)

    if input.sequence is not None:
        sequence = decode_sequence(input.sequence, tokenizers.sequence)

    plddt, ptm = None, None
    if input.structure is not None:
        # Note: We give priority to the structure tokens over the coordinates when decoding
        coordinates, plddt, ptm = decode_structure(
            structure_tokens=input.structure,
            structure_decoder=structure_token_decoder,
            structure_tokenizer=tokenizers.structure,
            sequence=sequence,
        )
    elif input.coordinates is not None:
        coordinates = input.coordinates[1:-1, ...]

    if input.secondary_structure is not None:
        secondary_structure = decode_secondary_structure(
            input.secondary_structure, tokenizers.secondary_structure
        )
    if input.sasa is not None:
        sasa = decode_sasa(input.sasa, tokenizers.sasa)
    if input.function is not None:
        if function_token_decoder is None:
            raise ValueError(
                "Cannot decode function annotations without a function token decoder"
            )
        function_track_annotations = decode_function_annotations(
            input.function,
            function_token_decoder=function_token_decoder,
            function_tokenizer=tokenizers.function,
        )
        function_annotations.extend(function_track_annotations)
    if input.residue_annotations is not None:
        residue_annotations = decode_residue_annotations(
            input.residue_annotations, tokenizers.residue_annotations
        )
        function_annotations.extend(residue_annotations)

    return ESMProtein(
        sequence=sequence,
        secondary_structure=secondary_structure,
        sasa=sasa,  # type: ignore
        function_annotations=function_annotations if function_annotations else None,
        coordinates=coordinates,
        plddt=plddt,
        ptm=ptm,
        potential_sequence_of_concern=input.potential_sequence_of_concern,
    )


def _bos_eos_warn(msg: str, tensor: torch.Tensor, tok: EsmTokenizerBase):
    if tensor[0] != tok.bos_token_id:
        warnings.warn(
            f"{msg} does not start with BOS token, token is ignored. BOS={tok.bos_token_id} vs {tensor}"
        )
    if tensor[-1] != tok.eos_token_id:
        warnings.warn(
            f"{msg} does not end with EOS token, token is ignored. EOS='{tok.eos_token_id}': {tensor}"
        )


def decode_sequence(
    sequence_tokens: torch.Tensor,
    sequence_tokenizer: EsmSequenceTokenizer,
    **kwargs,
) -> str:
    _bos_eos_warn("Sequence", sequence_tokens, sequence_tokenizer)
    sequence = sequence_tokenizer.decode(
        sequence_tokens,
        **kwargs,
    )
    sequence = sequence.replace(" ", "")
    sequence = sequence.replace(sequence_tokenizer.mask_token, C.MASK_STR_SHORT)
    sequence = sequence.replace(sequence_tokenizer.cls_token, "")
    sequence = sequence.replace(sequence_tokenizer.eos_token, "")

    return sequence


def decode_structure(
    structure_tokens: torch.Tensor,
    structure_decoder: StructureTokenDecoder,
    structure_tokenizer: StructureTokenizer,
    sequence: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    is_singleton = len(structure_tokens.size()) == 1
    if is_singleton:
        structure_tokens = structure_tokens.unsqueeze(0)
    else:
        raise ValueError(
            f"Only one structure can be decoded at a time, got structure tokens of shape {structure_tokens.size()}"
        )
    _bos_eos_warn("Structure", structure_tokens[0], structure_tokenizer)

    decoder_output = structure_decoder.decode(structure_tokens)
    bb_coords: torch.Tensor = decoder_output["bb_pred"][
        0, 1:-1, ...
    ]  # Remove BOS and EOS tokens
    bb_coords = bb_coords.detach().cpu()

    if "plddt" in decoder_output:
        plddt = decoder_output["plddt"][0, 1:-1]
        plddt = plddt.detach().cpu()
    else:
        plddt = None

    if "ptm" in decoder_output:
        ptm = decoder_output["ptm"]
    else:
        ptm = None

    chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
    chain = chain.infer_oxygen()
    return torch.tensor(chain.atom37_positions), plddt, ptm


def decode_secondary_structure(
    secondary_structure_tokens: torch.Tensor,
    ss_tokenizer: SecondaryStructureTokenizer,
) -> str:
    _bos_eos_warn("Secondary structure", secondary_structure_tokens, ss_tokenizer)
    secondary_structure_tokens = secondary_structure_tokens[1:-1]
    secondary_structure = ss_tokenizer.decode(
        secondary_structure_tokens,
    )
    return secondary_structure


def decode_sasa(
    sasa_tokens: torch.Tensor,
    sasa_tokenizer: SASADiscretizingTokenizer,
) -> list[float]:
    if sasa_tokens[0] != 0:
        raise ValueError("SASA does not start with 0 corresponding to BOS token")
    if sasa_tokens[-1] != 0:
        raise ValueError("SASA does not end with 0 corresponding to EOS token")
    sasa_tokens = sasa_tokens[1:-1]
    if sasa_tokens.dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ]:
        # Decode if int
        sasa = sasa_tokenizer.decode_float(sasa_tokens)
    else:
        # If already float, just convert to list
        sasa = sasa_tokens.tolist()

    return list_nan_to_none(sasa)


def decode_function_annotations(
    function_annotation_tokens: torch.Tensor,
    function_token_decoder: FunctionTokenDecoder,
    function_tokenizer: InterProQuantizedTokenizer,
    **kwargs,
) -> list[FunctionAnnotation]:
    # No need to check for BOS/EOS as function annotations are not affected

    function_annotations = decode_function_tokens(
        function_annotation_tokens,
        function_token_decoder=function_token_decoder,
        function_tokens_tokenizer=function_tokenizer,
        **kwargs,
    )
    return function_annotations


def decode_residue_annotations(
    residue_annotation_tokens: torch.Tensor,
    residue_annotation_decoder: ResidueAnnotationsTokenizer,
) -> list[FunctionAnnotation]:
    # No need to check for BOS/EOS as function annotations are not affected

    residue_annotations = decode_residue_annotation_tokens(
        residue_annotations_token_ids=residue_annotation_tokens,
        residue_annotations_tokenizer=residue_annotation_decoder,
    )
    return residue_annotations

def torsion_angles_to_frames(r: Affine3D,
            angles,
            aatype,
            rrgdf):
    rigid_type = type(r)
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]
    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = rigid_type.from_tensor(default_4x4)
    bb_rot = angles.new_zeros((*((1,) * len(angles.shape[:-1])), 2))
    bb_rot[..., 1] = 1
    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*angles.shape[:-2], -1, -1), angles], dim=-2)
    all_rots = alpha.new_zeros(default_r.shape + (4, 4))
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:3] = alpha
    all_rots = rigid_type.from_tensor(all_rots)
    all_frames = default_r.compose(all_rots)
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]
    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)
    # chi2_frame_to_bb.tensor_apply(lambda x: torch.unsqueeze(x, dim=-1))
    all_frames_to_bb = rigid_type.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,)
    
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    return all_frames_to_global

def frames_and_literature_positions_to_atom14_pos(
        r: Affine3D,
        aatype: torch.Tensor,
        default_frames,
        group_idx,
        atom_mask,
        lit_positions,):
    # [*, N, 14]
    group_mask = group_idx[aatype, ...]
    
    # [*, N, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )
    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :].mul(group_mask.bool())
    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.tensor_apply(lambda x: torch.sum(x, dim=-1))
    # [*, N, 14]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)
    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask
    return pred_positions

def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []
    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(residue_constants.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in residue_constants.atom_types
            ]
        )
        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )
    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)
    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein['aatype'].to(torch.long)
    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]
    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()
    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()
    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["aatype"].device
    )
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1
    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask
    return protein

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)
    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def atom14_to_atom37(atom14, protein):
    atom37_data = batched_gather(
        atom14,
        protein["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    atom37_data = atom37_data * protein["atom37_atom_exists"][..., None]
    atom37_data[~protein["atom37_atom_exists"].bool()] = float('inf')
    return atom37_data

def decode_structure_sidechain(
    structure_tokens: torch.Tensor,
    structure_decoder: StructureTokenDecoder,
    structure_tokenizer: StructureTokenizer,
    sequence: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    is_singleton = len(structure_tokens.size()) == 1
    if is_singleton:
        structure_tokens = structure_tokens.unsqueeze(0)
    else:
        raise ValueError(
            f"Only one structure can be decoded at a time, got structure tokens of shape {structure_tokens.size()}"
        )
    _bos_eos_warn("Structure", structure_tokens[0], structure_tokenizer)
    decoder_output = structure_decoder.decode(structure_tokens)
    bb_coords: torch.Tensor = decoder_output["bb_pred"][
        0, 1:-1, ...
    ]  # Remove BOS and EOS tokens
    bb_coords = bb_coords.detach().cpu()
    if "plddt" in decoder_output:
        plddt = decoder_output["plddt"][0, 1:-1]
        plddt = plddt.detach().cpu()
    else:
        plddt = None
    if "ptm" in decoder_output:
        ptm = decoder_output["ptm"]
    else:
        ptm = None
    backb_to_global = Affine3D.from_tensor(decoder_output['tensor7_affine'][:,1:-1])
    angles = decoder_output["angles"][:,1:-1]

    # backb_to_global = Affine3D.identity(backb_to_global)
    # fname = "/home/liuzijing/casp15/alphafold_output/Q92FR5_0/angles.pkl"
    # import pickle
    # with open(fname, 'rb') as f:
    #     xx1 = pickle.load(f)
    # angles = torch.tensor(xx1, dtype=angles.dtype, device=angles.device)

    aatype = residue_constants.sequence_to_onehot(sequence, residue_constants.restype_order_with_x)
    aatype = torch.tensor(
                    aatype[None,:],
                    dtype=torch.int64,
                    device=angles.device)
    aatype = torch.argmax(aatype, dim=-1)  # index (B,L)
    defualt_frame = torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=angles.dtype,
                    device=angles.device)
    all_frames_to_global = torsion_angles_to_frames(
        backb_to_global,
        angles,
        aatype, 
        defualt_frame)
    
    group_idx = torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=angles.device,
                    dtype=torch.int64)
    
    atom_mask = torch.tensor(
                    restype_atom14_mask,
                    dtype=angles.dtype,
                    device=angles.device)
    
    lit_positions = torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=angles.dtype,
                    device=angles.device)
    pred_xyz = frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
                defualt_frame,
                group_idx,
                atom_mask,
                lit_positions,
            )
    
    protein = {"aatype": aatype}
    
    protein = make_atom14_masks(protein)
    pred_xyz_atom37 = atom14_to_atom37(
            pred_xyz, protein)
    # chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
    # chain = chain.infer_oxygen()
    return pred_xyz_atom37.detach().cpu(), plddt, ptm