import torch
import torch.nn as nn
from attrdict import AttrDict
from typing import Tuple, Union
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig

from .progen.modeling_progen import ProGenForCausalLM,ProGenModel
from .progen.configuration_progen import ProGenConfig


class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(
        self, x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """

        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:  if it is a tuple of torch.Tensor,
                then it comes from the hybrid vision encoder, and x = high_res_x, low_res_x);
                otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """

        if isinstance(x_or_tuple, tuple):
            # self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else:
            x = x_or_tuple

        return self.layers(x)


class structure_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.structure_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.structure_token_embed, params.structure_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector
    elif "StructureTokenEncoder" in cls_name:
        from esm.models.vqvae import StructureTokenEncoder
        cls = StructureTokenEncoder
    elif "structure_head" in cls_name:
        cls = structure_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class StructureConfig(PretrainedConfig):
    model_type = "structure"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenStructureConfig(PretrainedConfig):
    model_type = "gen_structure"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    structure_config: StructureConfig
    aligner_config: AlignerConfig

    gen_structure_config: GenStructureConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: ProGenConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        structure_config = kwargs.get("structure_config", {})
        self.structure_config = StructureConfig(**structure_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_structure_config = kwargs.get("gen_structure_config", {})
        self.gen_structure_config = GenStructureConfig(**gen_structure_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, ProGenConfig):
            self.language_config = language_config
        else:
            self.language_config = ProGenConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ProGenModel):
            module.gradient_checkpointing = value


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        gen_structure_config = config.gen_structure_config

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_structure_config.params.structure_token_size, gen_structure_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = ProGenForCausalLM(language_config)

    def prepare_gen_img_embeds(self, structure_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(structure_ids))
    
    def forward(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        structure_seq_mask=None
    ):
        compute_dtype = self.language_model.dtype
        structure_embeds = self.prepare_gen_img_embeds(input_ids).to(torch.float32)
        seq_ids = input_ids.clone().detach()
        seq_ids[structure_seq_mask] = 0
        inputs_embeds = self.language_model.transformer.wte(seq_ids)
        inputs_embeds[structure_seq_mask] = structure_embeds[structure_seq_mask].to(compute_dtype)

        transformer_outputs = self.language_model.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs.last_hidden_state # B L d

        structure_logits = self.gen_head(hidden_states).to(torch.float32) # B L 4096
        lm_logits = structure_logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


AutoConfig.register("structure", StructureConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_structure", GenStructureConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)