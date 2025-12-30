import torch
from torch import nn
from typing import Optional, Union, Tuple, List, Dict, Any
from transformers import JambaConfig, JambaForCausalLM, JambaModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.models.base import BaseFamilyLitModule

class ProFamJambaConfig(JambaConfig):
    model_type = "profam_jamba"
    def __init__(
        self,
        max_segments: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_segments = max_segments

class ProFamJambaForCausalLM(JambaForCausalLM):
    config_class = ProFamJambaConfig
    
    def __init__(self, config: ProFamJambaConfig):
        super().__init__(config)
        # Add segment embeddings to the internal JambaModel
        self.model.segment_embedding = nn.Embedding(config.max_segments, config.hidden_size)
        
        # Initialize weights for segment embeddings
        self.model.segment_embedding.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
        if segment_ids is not None:
            inputs_embeds = inputs_embeds + self.model.segment_embedding(segment_ids)
            
        # Call the standard Jamba forward but with our custom inputs_embeds
        # We pass input_ids=None because we provide inputs_embeds
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

class HybridLitModule(BaseFamilyLitModule):
    def __init__(
        self,
        config: ProFamJambaConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        num_decay_steps: Optional[int] = None,
        scoring_max_tokens: int = 10240,
        use_kv_cache_for_scoring: bool = True,
        pass_res_pos_in_doc_as_position_ids: bool = True,
        optimizer: str = "adamw",
        override_optimizer_on_load: bool = False,
    ) -> None:
        
        model = ProFamJambaForCausalLM(config)
        
        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            scoring_max_tokens=scoring_max_tokens,
            use_kv_cache_for_scoring=use_kv_cache_for_scoring,
            override_optimizer_on_load=override_optimizer_on_load,
            pass_res_pos_in_doc_as_position_ids=pass_res_pos_in_doc_as_position_ids,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # We need to override training_step to pass segment_ids
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            segment_ids=batch.get("segment_ids")
        )
        loss = outputs.loss
        self.log_metrics(batch, outputs, "train", log_global=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if "DMS_scores" in batch:
            return self.validation_step_proteingym(batch)
        
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            segment_ids=batch.get("segment_ids")
        )
        loss = outputs.loss
        self.log_metrics(batch, outputs, "val", log_global=dataloader_idx == 0)
        return loss
