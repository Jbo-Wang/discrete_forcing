# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        
        if temb.dim() == 3:
            # temb: (B, seq_len, 2*embedding_dim) - per-position conditioning
            scale, shift = temb.chunk(2, dim=-1)
            x = self.norm(x) * (1 + scale) + shift
        else:
            # temb: (B, 2*embedding_dim) - shared conditioning for all positions
            scale, shift = temb.chunk(2, dim=1)
            x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1( 
            norm_hidden_states, 
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class HybridDiT(ModelMixin, ConfigMixin):
    """Hybrid DiT with shared layers and separate branches for continuous/discrete actions.
    
    Architecture:
    - First num_shared_layers (default 4) are shared across both modalities
    - Remaining layers are split into two parallel branches:
      - continuous_transformer_blocks: for continuous action generation
      - discrete_transformer_blocks: for discrete action generation
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        num_shared_layers: int = 10,  # Number of shared layers before splitting
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        branch_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.branch_inner_dim = branch_hidden_dim or self.inner_dim
        if self.branch_inner_dim % self.config.attention_head_dim != 0:
            raise ValueError(
                "branch_hidden_dim must be divisible by attention_head_dim, "
                f"got branch_hidden_dim={self.branch_inner_dim}, "
                f"attention_head_dim={self.config.attention_head_dim}."
            )
        self.branch_num_attention_heads = self.branch_inner_dim // self.config.attention_head_dim
        self.gradient_checkpointing = False
        
        # Validate num_shared_layers
        assert num_shared_layers < num_layers, "num_shared_layers must be less than num_layers"
        self.num_shared_layers = num_shared_layers
        self.num_branch_layers = num_layers - num_shared_layers

        # Timestep encoder
        compute_dtype = getattr(self.config, 'compute_dtype', torch.float32)
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )
        self.continuous_timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )
        self.discrete_timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

        # Build shared transformer blocks (first num_shared_layers)
        shared_blocks = []
        for idx in range(self.num_shared_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            shared_blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
        self.shared_transformer_blocks = nn.ModuleList(shared_blocks)

        if self.branch_inner_dim == self.inner_dim:
            self.continuous_branch_in_proj = nn.Identity()
            self.discrete_branch_in_proj = nn.Identity()
            self.continuous_branch_out_proj = nn.Identity()
            self.discrete_branch_out_proj = nn.Identity()
            self.continuous_branch_temb_proj = nn.Identity()
            self.discrete_branch_temb_proj = nn.Identity()
        else:
            self.continuous_branch_in_proj = nn.Linear(self.inner_dim, self.branch_inner_dim)
            self.discrete_branch_in_proj = nn.Linear(self.inner_dim, self.branch_inner_dim)
            self.continuous_branch_out_proj = nn.Linear(self.branch_inner_dim, self.inner_dim)
            self.discrete_branch_out_proj = nn.Linear(self.branch_inner_dim, self.inner_dim)
            self.continuous_branch_temb_proj = nn.Linear(self.inner_dim, self.branch_inner_dim)
            self.discrete_branch_temb_proj = nn.Linear(self.inner_dim, self.branch_inner_dim)

        # Build continuous and discrete branch transformer blocks
        continuous_blocks = []
        discrete_blocks = []
        for idx in range(self.num_branch_layers):
            global_idx = self.num_shared_layers + idx
            use_self_attn = global_idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            continuous_blocks.append(
                BasicTransformerBlock(
                    self.branch_inner_dim,
                    self.branch_num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
            discrete_blocks.append(
                BasicTransformerBlock(
                    self.branch_inner_dim,
                    self.branch_num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
        self.continuous_transformer_blocks = nn.ModuleList(continuous_blocks)
        self.discrete_transformer_blocks = nn.ModuleList(discrete_blocks)

        # Output blocks for each branch
        # Continuous branch output
        self.continuous_norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.continuous_proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.continuous_proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        
        # Discrete branch output
        self.discrete_norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.discrete_proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.discrete_proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        
        print(
            "Total number of HybridDiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        return_branch_outputs: str = "both",  # "continuous", "discrete", or "both"
    ):
        """
        Args:
            hidden_states: Input tensor (B, T, D)
            encoder_hidden_states: Cross-attention conditioning (B, S, D)
            timestep: Timestep for conditioning
            return_all_hidden_states: Whether to return all intermediate hidden states
            encoder_attention_mask: Attention mask for encoder_hidden_states
            return_branch_outputs: Which branch outputs to return ("continuous", "discrete", or "both")
        Returns:
            If return_branch_outputs="both": (continuous_output, discrete_output)
            If return_branch_outputs="continuous": continuous_output
            If return_branch_outputs="discrete": discrete_output
        """
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through shared transformer blocks
        for idx, block in enumerate(self.shared_transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Split into two branches
        continuous_hidden = self.continuous_branch_in_proj(hidden_states)
        discrete_hidden = self.discrete_branch_in_proj(hidden_states.clone())
        continuous_temb = self.continuous_branch_temb_proj(temb)
        discrete_temb = self.discrete_branch_temb_proj(temb)

        # Process through continuous branch
        continuous_all_hidden = [continuous_hidden]
        for idx, block in enumerate(self.continuous_transformer_blocks):
            global_idx = self.num_shared_layers + idx
            if global_idx % 2 == 1 and self.config.interleave_self_attention:
                continuous_hidden = block(
                    continuous_hidden,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=continuous_temb,
                )
            else:
                continuous_hidden = block(
                    continuous_hidden,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=continuous_temb,
                )
            continuous_all_hidden.append(continuous_hidden)

        # Process through discrete branch
        discrete_all_hidden = [discrete_hidden]
        for idx, block in enumerate(self.discrete_transformer_blocks):
            global_idx = self.num_shared_layers + idx
            if global_idx % 2 == 1 and self.config.interleave_self_attention:
                discrete_hidden = block(
                    discrete_hidden,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=discrete_temb,
                )
            else:
                discrete_hidden = block(
                    discrete_hidden,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=discrete_temb,
                )
            discrete_all_hidden.append(discrete_hidden)

        continuous_hidden = self.continuous_branch_out_proj(continuous_hidden)
        discrete_hidden = self.discrete_branch_out_proj(discrete_hidden)
        
        # Output processing for continuous branch
        continuous_shift, continuous_scale = self.continuous_proj_out_1(F.silu(temb)).chunk(2, dim=1)
        continuous_hidden = self.continuous_norm_out(continuous_hidden) * (1 + continuous_scale[:, None]) + continuous_shift[:, None]
        continuous_output = self.continuous_proj_out_2(continuous_hidden)

        # Output processing for discrete branch
        discrete_shift, discrete_scale = self.discrete_proj_out_1(F.silu(temb)).chunk(2, dim=1)
        discrete_hidden = self.discrete_norm_out(discrete_hidden) * (1 + discrete_scale[:, None]) + discrete_shift[:, None]
        discrete_output = self.discrete_proj_out_2(discrete_hidden)

        if return_all_hidden_states:
            if return_branch_outputs == "continuous":
                return continuous_output, all_hidden_states + continuous_all_hidden
            elif return_branch_outputs == "discrete":
                return discrete_output, all_hidden_states + discrete_all_hidden
            else:  # "both"
                return (continuous_output, discrete_output), all_hidden_states + continuous_all_hidden + discrete_all_hidden
        else:
            if return_branch_outputs == "continuous":
                return continuous_output
            elif return_branch_outputs == "discrete":
                return discrete_output
            else:  # "both"
                return continuous_output, discrete_output


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
