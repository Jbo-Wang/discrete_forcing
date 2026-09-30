"""Qwen3-VL QwenPILF with the compressed, layer-wise QwenPI_v3 interface."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.HybridDiscreteContinuous_ActionHeader_up import (
    HybridLayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("QwenPILF_v3")
class QwenPILFv3(baseframework):
    """Hybrid discrete/continuous action expert over projected VLM layers."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=config)

        hf_config = self.qwen_vl_interface.model.config
        text_config = getattr(hf_config, "text_config", hf_config)
        llm_hidden_size = int(text_config.hidden_size)
        num_vl_layers = int(text_config.num_hidden_layers)
        config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        config.framework.qwenvl.num_vl_layers = num_vl_layers

        diffusion_cfg = config.framework.action_model.diffusion_model_cfg
        configured_dit_hidden_dim = diffusion_cfg.get("action_dit_hidden_dim", 1024)
        action_dit_hidden_dim = int(
            llm_hidden_size
            if configured_dit_hidden_dim is None
            else configured_dit_hidden_dim
        )
        diffusion_cfg.action_dit_hidden_dim = action_dit_hidden_dim

        self.action_model: HybridLayerwiseFlowmatchingActionHead = get_action_model(
            config=config
        )
        num_action_layers = (
            len(self.action_model.model.shared_transformer_blocks)
            + len(self.action_model.model.continuous_transformer_blocks)
        )
        if num_action_layers != num_vl_layers:
            raise ValueError(
                f"Action/VLM layer mismatch: action={num_action_layers}, VLM={num_vl_layers}."
            )

        self.project_layers = nn.ModuleList(
            [
                nn.Identity()
                if llm_hidden_size == action_dit_hidden_dim
                else nn.Sequential(
                    nn.LayerNorm(llm_hidden_size),
                    nn.Linear(llm_hidden_size, action_dit_hidden_dim),
                )
                for _ in range(num_action_layers)
            ]
        )

        trainer_cfg = config.trainer
        enable_gc = bool(
            getattr(
                trainer_cfg,
                "gradient_checkpointing",
                getattr(trainer_cfg, "enable_gradient_checkpointing", False),
            )
        )
        if enable_gc:
            if hasattr(self.qwen_vl_interface.model, "gradient_checkpointing_enable"):
                self.qwen_vl_interface.model.gradient_checkpointing_enable()
            if hasattr(self.action_model.model, "gradient_checkpointing"):
                self.action_model.model.gradient_checkpointing = True

        action_cfg = config.framework.action_model
        if hasattr(action_cfg, "action_horizon"):
            self.action_horizon = int(action_cfg.action_horizon)
        else:
            self.action_horizon = int(action_cfg.future_action_window_size) + 1
        self.discrete_loss_weight = float(
            getattr(action_cfg, "discrete_loss_weight", 1.0)
        )
        if (
            not np.isfinite(self.discrete_loss_weight)
            or self.discrete_loss_weight < 0.0
        ):
            raise ValueError(
                "discrete_loss_weight must be a finite non-negative value, got "
                f"{self.discrete_loss_weight}."
            )

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            num_layers = len(self.project_layers)
            vl_hidden_states = list(outputs.hidden_states[-num_layers:])
            if len(vl_hidden_states) != num_layers:
                raise ValueError(
                    f"Expected {num_layers} VLM hidden states, got {len(vl_hidden_states)}."
                )
            projected = [
                projector(hidden)
                for projector, hidden in zip(self.project_layers, vl_hidden_states)
            ]
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        return projected, attention_mask

    def forward(
        self, examples: List[dict] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        vl_embs_list, attention_mask = self._encode_vl_hidden_states(
            batch_images, instructions
        )
        base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_targets = torch.as_tensor(
                np.asarray(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )[:, -self.action_horizon :, :]
            discrete_targets = self.action_model.discretize_actions(action_targets)
            repeated_steps = int(
                getattr(self.config.framework.action_model, "repeated_diffusion_steps", 2)
            )
            original_batch_size = action_targets.shape[0]
            action_targets = action_targets.repeat(repeated_steps, 1, 1)
            discrete_targets = discrete_targets.repeat(repeated_steps, 1)
            repeated_vl_embs = [
                hidden.repeat(repeated_steps, 1, 1) for hidden in vl_embs_list
            ]
            repeated_mask = (
                attention_mask.repeat(repeated_steps, 1)
                if attention_mask is not None
                else None
            )
            phase_is_continuous = None
            if bool(
                getattr(
                    self.config.framework.action_model,
                    "paired_phase_sampling",
                    False,
                )
            ):
                if repeated_steps < 2 or repeated_steps % 2 != 0:
                    raise ValueError(
                        "paired_phase_sampling requires an even repeated_diffusion_steps >= 2, "
                        f"got {repeated_steps}."
                    )
                repeat_ids = torch.arange(
                    repeated_steps, device=base_hidden.device
                ).repeat_interleave(original_batch_size)
                phase_is_continuous = repeat_ids >= (repeated_steps // 2)

            continuous_loss, discrete_loss = self.action_model(
                repeated_vl_embs,
                action_targets,
                discrete_actions=discrete_targets,
                state=None,
                encoder_attention_mask=repeated_mask,
                phase_is_continuous=phase_is_continuous,
            )

        return {
            "action_loss": (
                continuous_loss + self.discrete_loss_weight * discrete_loss
            ),
            "continuous_loss": continuous_loss,
            "discrete_loss": discrete_loss,
        }

    @torch.inference_mode()
    def predict_action(
        self, examples: List[dict] = None, **kwargs
    ) -> Dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        target_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if target_size is None:
            target_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if target_size:
            batch_images = resize_images(batch_images, target_size=target_size)

        vl_embs_list, attention_mask = self._encode_vl_hidden_states(
            batch_images, instructions
        )
        generator = None
        if bool(kwargs.get("use_ddim", False)) or kwargs.get("do_sample", True) is False:
            generator = torch.Generator()
            generator.manual_seed(
                int(kwargs.get("sampling_seed", getattr(self.config, "seed", 0)))
            )

        with torch.autocast("cuda", dtype=torch.float32):
            outputs = self.action_model.predict_action(
                vl_embs_list,
                state=None,
                num_inference_timesteps=kwargs.get("num_inference_timesteps"),
                num_discrete_steps=kwargs.get("num_discrete_steps"),
                num_continuous_steps=kwargs.get("num_continuous_steps"),
                generator=generator,
                discrete_rollout_mode=kwargs.get("discrete_rollout_mode"),
                encoder_attention_mask=attention_mask,
            )

        return {
            "normalized_actions": outputs["continuous_actions"]
            .detach()
            .float()
            .cpu()
            .numpy(),
            "discrete_normalized_actions": outputs[
                "discrete_actions_continuous"
            ].detach().float().cpu().numpy(),
            "discrete_rollout_mode": outputs.get("discrete_rollout_mode"),
        }
