# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from starVLA.model.modules.action_model.flow_matching_head.action_encoder import SinusoidalPositionalEncoding, swish
from starVLA.model.modules.action_model.flow_matching_head.hybrid_cross_attention_dit_up import HybridDiT, SelfAttentionTransformer

class CategorySpecificLinear(nn.Module):

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

class CategorySpecificMLP(nn.Module):

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))

class ContinuousActionEncoder(nn.Module):

    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError('Expected `timesteps` to have shape (B,) so we can replicate across T.')
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x

class ScalarContinuousActionEncoder(nn.Module):

    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.value_proj = nn.Linear(1, hidden_size)
        self.dim_embedding = nn.Embedding(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)
        nn.init.normal_(self.dim_embedding.weight, mean=0.0, std=0.02)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T * action_dim, hidden_size)
        """
        B, T, D = actions.shape
        if D != self.action_dim:
            raise ValueError(f'Expected action_dim={self.action_dim}, got {D}.')
        seq_len = T * D
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        else:
            raise ValueError('Expected `timesteps` to have shape (B,) so we can replicate across T * action_dim.')
        actions_flat = actions.reshape(B, seq_len, 1)
        a_emb = self.value_proj(actions_flat)
        dim_ids = torch.arange(D, device=actions.device, dtype=torch.long).repeat(T)
        dim_emb = self.dim_embedding(dim_ids).unsqueeze(0).expand(B, -1, -1)
        a_emb = a_emb + dim_emb.to(dtype=a_emb.dtype)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x

class DiscreteActionEncoder(nn.Module):

    def __init__(self, num_discrete_bins, hidden_size=1024):
        """
        Args:
            num_discrete_bins: Number of discrete bins (e.g., 256)
            hidden_size: Output hidden dimension
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_discrete_bins = num_discrete_bins
        self.layer1 = nn.Embedding(num_discrete_bins + 1, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, L) -- integer indices in [0, num_discrete_bins], L = T * action_dim (flattened)
        timesteps: shape (B,)  -- a single scalar per batch item (noise level for discrete denoising)
        returns:   shape (B, L, hidden_size)
        """
        B, L = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, L)
        else:
            raise ValueError('Expected `timesteps` to have shape (B,) so we can replicate across L.')
        a_emb = self.layer1(actions.long())
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x

class SharedActionTokenFusion(nn.Module):

    def __init__(self, hidden_size=1024):
        super().__init__()
        self.layer1 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, continuous_features, discrete_features):
        x = torch.cat([continuous_features, discrete_features], dim=-1)
        x = swish(self.layer1(x))
        x = self.layer2(x)
        return x

class MultiEmbodimentContinuousActionEncoder(nn.Module):

    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError('Expected `timesteps` to have shape (B,) so we can replicate across T.')
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        return x

@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """Configuration for the hybrid discrete/continuous action head."""
    add_pos_embed: bool = field(default=True, metadata={'help': 'Whether to add positional embedding'})
    diffusion_model_cfg: dict = field(default=None, metadata={'help': 'Diffusion model configuration.'})
    input_embedding_dim: int = field(default=1536, metadata={'help': 'Input embedding channel dimension.'})
    hidden_size: int = field(default=1024, metadata={'help': 'Input embedding dimension.'})
    max_seq_len: int = field(default=1024, metadata={'help': 'Maxium Sequence Length'})
    action_dim: int = field(default=None, metadata={'help': 'Action dimension.'})
    action_horizon: int = field(default=None, metadata={'help': 'Action horizon.'})
    discrete_action_horizon: int = field(default=None, metadata={'help': 'Number of coarse time steps modeled by the discrete branch.'})
    discrete_projection_ridge: float = field(default=0.0001, metadata={'help': 'Ridge regularization for projecting full actions onto coarse knots.'})
    noise_beta_alpha: float = field(default=1.5, metadata={'help': ''})
    noise_beta_beta: float = field(default=1.0, metadata={'help': ''})
    noise_s: float = field(default=0.999, metadata={'help': 'Flow matching noise Beta distribution s.'})
    num_timestep_buckets: int = field(default=1000, metadata={'help': 'Number of timestep discretization buckets.'})
    share_action_tokens_by_dim: bool = field(default=False, metadata={'help': 'If True, continuous and discrete branches share one action-token stream with one token per (time step, action dim) slot.'})
    discrete_rollout_mode: str = field(default='topk_lock', metadata={'help': 'Discrete inference rollout mode. `topk_lock` keeps the current confidence-based progressive reveal, `refresh_all` refreshes all discrete tokens every step without early locking.'})
    num_inference_timesteps: int = field(default=None, metadata={'help': 'Number of inference steps for noise diffusion.'})
    continuous_condition_mask_ratio: float = field(default=0.0, metadata={'help': 'Mask ratio applied to GT discrete tokens for continuous-branch conditioning during training.'})
    continuous_condition_use_source_alpha: bool = field(default=False, metadata={'help': 'If True, continuous-phase discrete conditioning keeps a continuous_source_alpha fraction of GT tokens and replaces the rest with different valid random bins.'})
    continuous_source_continuous_phase_only: bool = field(default=False, metadata={'help': 'If True, apply the discrete-guided continuous source only to continuous-phase samples and keep the discrete-phase continuous stream as pure Gaussian noise. False preserves the legacy behavior.'})
    continuous_one_step_loss_prob: float = field(default=0.0, metadata={'help': 'For continuous-half samples, probability of applying an auxiliary one-step refinement loss from the current continuous source toward the GT action.'})
    continuous_one_step_loss_weight: float = field(default=0.0, metadata={'help': 'Weight of the auxiliary one-step continuous refinement loss. Set to 0 to disable the loss.'})
    branch_communication_on_self_attention: bool = field(default=False, metadata={'help': 'If True, continuous and discrete branch tokens communicate on branch self-attention layers by running the self-attention block over concatenated branch tokens.'})
    branch_input_mode: str = field(default='full', metadata={'help': 'Tokens passed to each non-shared branch after the shared layers. `full` copies the complete sequence to both branches; `action_only` passes only the corresponding action tokens; `future_action` passes future tokens plus the corresponding action tokens.'})
    isolate_branch_tokens_before_shared: bool = field(default=False, metadata={'help': 'If True, remove discrete tokens from the continuous path and continuous tokens from the discrete path before the shared transformer layers.'})
    paired_phase_sampling: bool = field(default=False, metadata={'help': 'Assign repeated copies of every trajectory evenly to the discrete and continuous phases.'})
    disjoint_branch_batch_routing: bool = field(default=False, metadata={'help': 'After the shared layers, run each branch only on the samples supervised by that branch.'})
    continuous_condition_skip_gripper: bool = field(default=True, metadata={'help': 'Whether to avoid perturbing the gripper token positions during continuous-half conditioning.'})
    joint_branch_supervision: bool = field(default=False, metadata={'help': 'If True, optimize continuous and discrete losses on every sample instead of splitting the batch into discrete-only and continuous-only halves.'})
    discrete_loss_on_all_tokens: bool = field(default=False, metadata={'help': 'If True, compute discrete CE on all token positions. If False, keep the old masked-token-only CE.'})
    discrete_visible_loss_weight: float = field(default=0.0, metadata={'help': 'Extra weight applied to CE on originally visible discrete tokens. The effective discrete loss becomes masked_ce + weight * visible_ce.'})
    discrete_loss_weight: float = field(default=1.0, metadata={'help': 'Multiplier applied to the reduced discrete loss when it is combined with the continuous loss by QwenPILF_v3.'})
    max_num_embodiments: int = field(default=32, metadata={'help': 'Number of embodiments.'})
    tune_projector: bool = field(default=True, metadata={'help': 'Whether to tune the projector.'})
    tune_diffusion_model: bool = field(default=True, metadata={'help': 'Whether to tune the diffusion model.'})
    load_pretrained_det_decode_layer_path: str = field(default=None, metadata={'help': 'Path to pretrained detection model.'})
    detection_coeff: float = field(default=1.0, metadata={'help': 'Detection coefficient.'})
    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={'help': 'Number of target vision tokens.'})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class HybridLayerwiseFlowmatchingActionHead(nn.Module):

    def __init__(self, global_config, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg
        configured_dit_hidden_dim = diffusion_model_cfg.get('action_dit_hidden_dim', None)
        dit_hidden_dim = int(global_config.framework.qwenvl.vl_hidden_dim if configured_dit_hidden_dim is None else configured_dit_hidden_dim)
        resolved_diffusion_cfg = {key: diffusion_model_cfg[key] for key in diffusion_model_cfg if key != 'action_dit_hidden_dim'}
        attention_head_dim = int(resolved_diffusion_cfg.get('attention_head_dim', 64))
        if dit_hidden_dim % attention_head_dim != 0:
            raise ValueError(f'action_dit_hidden_dim={dit_hidden_dim} must be divisible by attention_head_dim={attention_head_dim}.')
        resolved_diffusion_cfg.update(num_layers=int(global_config.framework.qwenvl.num_vl_layers), input_embedding_dim=dit_hidden_dim, output_dim=dit_hidden_dim, cross_attention_dim=dit_hidden_dim, num_attention_heads=dit_hidden_dim // attention_head_dim)
        self.input_embedding_dim = dit_hidden_dim
        self.model = HybridDiT(**resolved_diffusion_cfg)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.continuous_dim = action_config.action_dim
        if hasattr(action_config, 'action_horizon'):
            self.continuous_horizon = int(action_config.action_horizon)
        else:
            self.continuous_horizon = int(action_config.future_action_window_size) + 1
        self.share_action_tokens_by_dim = bool(getattr(action_config, 'share_action_tokens_by_dim', False))
        self.discrete_rollout_mode = self._validate_discrete_rollout_mode(getattr(action_config, 'discrete_rollout_mode', 'topk_lock'))
        self.default_num_discrete_steps = getattr(action_config, 'num_discrete_steps', None)
        self.default_num_continuous_steps = getattr(action_config, 'num_continuous_steps', None)
        configured_discrete_horizon = getattr(action_config, 'discrete_action_horizon', None)
        self.discrete_action_horizon = int(self.continuous_horizon if configured_discrete_horizon is None else configured_discrete_horizon)
        if not 2 <= self.discrete_action_horizon <= self.continuous_horizon:
            raise ValueError(f'discrete_action_horizon must be between 2 and action_horizon, got {self.discrete_action_horizon} and {self.continuous_horizon}.')
        if self.share_action_tokens_by_dim and self.discrete_action_horizon != self.continuous_horizon:
            raise ValueError('share_action_tokens_by_dim requires matching continuous and discrete horizons.')
        self.discrete_projection_ridge = float(getattr(action_config, 'discrete_projection_ridge', 0.0001))
        if self.discrete_projection_ridge < 0.0:
            raise ValueError('discrete_projection_ridge must be non-negative.')
        interpolation_matrix = self._build_linear_interpolation_matrix(source_horizon=self.discrete_action_horizon, target_horizon=self.continuous_horizon)
        if self.discrete_action_horizon == self.continuous_horizon:
            projection_matrix = torch.eye(self.continuous_horizon, dtype=interpolation_matrix.dtype)
        else:
            gram = interpolation_matrix.transpose(0, 1) @ interpolation_matrix
            gram = gram + self.discrete_projection_ridge * torch.eye(self.discrete_action_horizon, dtype=interpolation_matrix.dtype)
            projection_matrix = torch.linalg.solve(gram, interpolation_matrix.transpose(0, 1))
        self.register_buffer('discrete_interpolation_matrix', interpolation_matrix, persistent=False)
        self.register_buffer('discrete_projection_matrix', projection_matrix, persistent=False)
        self.discrete_dim = 1
        self.discrete_horizon = self.discrete_action_horizon * action_config.action_dim
        self.num_inference_timesteps = action_config.num_inference_timesteps
        self.continuous_source_mode = self._validate_continuous_source_mode(getattr(action_config, 'continuous_source_mode', 'gaussian'))
        self.continuous_source_alpha = self._validate_probability(getattr(action_config, 'continuous_source_alpha', 0.0), 'continuous_source_alpha')
        self.continuous_source_continuous_phase_only = bool(getattr(action_config, 'continuous_source_continuous_phase_only', False))
        self.continuous_condition_mask_ratio = self._validate_probability(getattr(action_config, 'continuous_condition_mask_ratio', 0.0), 'continuous_condition_mask_ratio')
        self.continuous_condition_use_source_alpha = bool(getattr(action_config, 'continuous_condition_use_source_alpha', False))
        self.continuous_one_step_loss_prob = self._validate_probability(getattr(action_config, 'continuous_one_step_loss_prob', 0.0), 'continuous_one_step_loss_prob')
        self.continuous_one_step_loss_weight = max(0.0, float(getattr(action_config, 'continuous_one_step_loss_weight', 0.0)))
        self.branch_communication_on_self_attention = bool(getattr(action_config, 'branch_communication_on_self_attention', False))
        self.branch_input_mode = self._validate_branch_input_mode(getattr(action_config, 'branch_input_mode', 'full'))
        self.isolate_branch_tokens_before_shared = bool(getattr(action_config, 'isolate_branch_tokens_before_shared', False))
        self.paired_phase_sampling = bool(getattr(action_config, 'paired_phase_sampling', False))
        self.disjoint_branch_batch_routing = bool(getattr(action_config, 'disjoint_branch_batch_routing', False))
        if self.branch_input_mode != 'full' and self.branch_communication_on_self_attention:
            raise ValueError('Reduced branch inputs are not compatible with branch communication; disable branch communication.')
        if self.disjoint_branch_batch_routing and self.branch_communication_on_self_attention:
            raise ValueError('disjoint_branch_batch_routing is not compatible with branch communication.')
        if self.disjoint_branch_batch_routing and (not self.paired_phase_sampling):
            raise ValueError('disjoint_branch_batch_routing requires paired_phase_sampling=true.')
        if self.disjoint_branch_batch_routing and bool(getattr(action_config, 'joint_branch_supervision', False)):
            raise ValueError('disjoint_branch_batch_routing requires joint_branch_supervision=false.')
        if self.isolate_branch_tokens_before_shared:
            if self.share_action_tokens_by_dim:
                raise ValueError('isolate_branch_tokens_before_shared requires separate action tokens.')
            if self.branch_input_mode != 'full':
                raise ValueError('isolate_branch_tokens_before_shared requires branch_input_mode=full.')
            if self.branch_communication_on_self_attention:
                raise ValueError('isolate_branch_tokens_before_shared is incompatible with branch communication.')
            if not self.disjoint_branch_batch_routing:
                raise ValueError('isolate_branch_tokens_before_shared requires disjoint_branch_batch_routing=true.')
        self.continuous_condition_skip_gripper = bool(getattr(action_config, 'continuous_condition_skip_gripper', True))
        self.discrete_visible_loss_weight = max(0.0, float(getattr(action_config, 'discrete_visible_loss_weight', 0.0)))
        self.state_encoder = MLP(input_dim=action_config.state_dim, output_dim=self.input_embedding_dim) if action_config.state_dim else None
        if self.share_action_tokens_by_dim:
            self.continuous_encoder = ScalarContinuousActionEncoder(action_dim=action_config.action_dim, hidden_size=self.input_embedding_dim)
            self.shared_action_fusion = SharedActionTokenFusion(hidden_size=self.input_embedding_dim)
            continuous_decoder_output_dim = 1
        else:
            self.continuous_encoder = ContinuousActionEncoder(action_dim=action_config.action_dim, hidden_size=self.input_embedding_dim)
            self.shared_action_fusion = None
            continuous_decoder_output_dim = self.continuous_dim
        self.continuous_decoder = MLP(input_dim=self.input_embedding_dim, hidden_dim=1024, output_dim=continuous_decoder_output_dim)
        self.num_discrete_bins = getattr(action_config, 'num_discrete_bins', 256)
        self.discrete_encoder = DiscreteActionEncoder(num_discrete_bins=self.num_discrete_bins, hidden_size=self.input_embedding_dim)
        self.discrete_decoder = MLP(input_dim=self.input_embedding_dim, hidden_dim=1024, output_dim=self.num_discrete_bins)
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    @staticmethod
    def _validate_probability(probability, name: str) -> float:
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f'{name} must be in [0, 1], got {probability}.')
        return probability

    @staticmethod
    def _sample_other_bins(gt_tokens: torch.Tensor, num_bins: int) -> torch.Tensor:
        if num_bins < 2:
            raise ValueError('num_bins must be at least 2 to sample a different bin.')
        sampled = torch.randint(0, num_bins - 1, gt_tokens.shape, device=gt_tokens.device, dtype=torch.long)
        return sampled + (sampled >= gt_tokens.long()).long()

    @staticmethod
    def _build_linear_interpolation_matrix(source_horizon: int, target_horizon: int) -> torch.Tensor:
        """Build the fixed linear basis used by both training and inference."""
        positions = torch.linspace(0, source_horizon - 1, target_horizon, dtype=torch.float32)
        left = positions.floor().long()
        right = (left + 1).clamp(max=source_horizon - 1)
        right_weight = positions - left.to(dtype=positions.dtype)
        left_weight = 1.0 - right_weight
        matrix = torch.zeros(target_horizon, source_horizon, dtype=torch.float32)
        rows = torch.arange(target_horizon)
        matrix[rows, left] += left_weight
        matrix[rows, right] += right_weight
        return matrix

    @staticmethod
    def _validate_discrete_rollout_mode(mode: str) -> str:
        mode = str(mode)
        allowed = {'topk_lock', 'refresh_all'}
        if mode not in allowed:
            raise ValueError(f'discrete_rollout_mode must be one of {sorted(allowed)}, got {mode}.')
        return mode

    @staticmethod
    def _validate_continuous_source_mode(mode: str) -> str:
        mode = str(mode)
        allowed = {'gaussian', 'mix_discrete'}
        if mode not in allowed:
            raise ValueError(f'continuous_source_mode must be one of {sorted(allowed)}, got {mode}.')
        return mode

    @staticmethod
    def _validate_branch_input_mode(mode: str) -> str:
        mode = str(mode)
        allowed = {'full', 'action_only', 'future_action'}
        if mode not in allowed:
            raise ValueError(f'branch_input_mode must be one of {sorted(allowed)}, got {mode}.')
        return mode

    def _mask_discrete_tokens(self, discrete_tokens: torch.Tensor, mask_ratio: float):
        mask_ratio = self._validate_probability(mask_ratio, 'mask_ratio')
        masked_tokens = discrete_tokens.clone()
        mask_positions = torch.zeros_like(discrete_tokens, dtype=torch.bool)
        if mask_ratio <= 0.0:
            return (masked_tokens, mask_positions)
        batch_size, seq_len = masked_tokens.shape
        num_mask = int(round(seq_len * mask_ratio))
        if num_mask <= 0:
            return (masked_tokens, mask_positions)
        num_mask = min(seq_len, num_mask)
        for batch_idx in range(batch_size):
            selected_idx = torch.randperm(seq_len, device=masked_tokens.device)[:num_mask]
            masked_tokens[batch_idx, selected_idx] = self.num_discrete_bins
            mask_positions[batch_idx, selected_idx] = True
        return (masked_tokens, mask_positions)

    def discretize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Project full actions to coarse knots and discretize them into scalar tokens.
        
        Args:
            actions: Continuous actions with shape (B, T, D) where:
                     B = batch size
                     T = action horizon (time steps)
                     D = action dimension
        
        Returns:
            Discrete indices with shape (B, discrete_action_horizon * D).
        """
        B, T, D = actions.shape
        if T != self.continuous_horizon or D != self.continuous_dim:
            raise ValueError(f'Expected actions shaped [B, {self.continuous_horizon}, {self.continuous_dim}], got {tuple(actions.shape)}.')
        projection = self.discrete_projection_matrix.to(device=actions.device, dtype=torch.float32)
        coarse_actions = torch.einsum('kt,btd->bkd', projection, actions.float()).clamp(min=-1.0, max=1.0)
        actions_flat = coarse_actions.reshape(B, -1)
        actions_normalized = (actions_flat + 1.0) / 2.0
        actions_normalized = torch.clamp(actions_normalized, 0.0, 1.0)
        discrete_indices = (actions_normalized * (self.num_discrete_bins - 1)).long()
        return discrete_indices

    def decoded_discrete_action(self, discrete_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode bin indices into normalized continuous action values.
        
        Args:
            discrete_indices: shape (B, T*D) or (B, T, D), with bin indices
                in [0, num_discrete_bins-1].
            
        Returns:
            Actions with the same shape as the input and values in [-1, 1].
        """
        indices_float = discrete_indices.float()
        normalized = indices_float / (self.num_discrete_bins - 1)
        actions = normalized * 2.0 - 1.0
        return actions

    def _decode_discrete_tokens_to_actions(self, discrete_tokens: torch.Tensor) -> torch.Tensor:
        discrete_tokens = discrete_tokens.clamp(min=0, max=self.num_discrete_bins - 1)
        coarse_actions = self.decoded_discrete_action(discrete_tokens).view(discrete_tokens.shape[0], self.discrete_action_horizon, self.continuous_dim)
        interpolation = self.discrete_interpolation_matrix.to(device=coarse_actions.device, dtype=torch.float32)
        full_actions = torch.einsum('tk,bkd->btd', interpolation, coarse_actions.float())
        return full_actions.to(dtype=coarse_actions.dtype)

    def _build_continuous_source_actions(self, noise_actions: torch.Tensor, discrete_tokens: torch.Tensor | None=None) -> torch.Tensor:
        if self.continuous_source_mode == 'gaussian' or discrete_tokens is None:
            return noise_actions
        decoded_actions = self._decode_discrete_tokens_to_actions(discrete_tokens)
        decoded_actions = decoded_actions.to(device=noise_actions.device, dtype=noise_actions.dtype)
        return (1.0 - self.continuous_source_alpha) * noise_actions + self.continuous_source_alpha * decoded_actions

    def sample_time(self, batch_size, device, dtype, phase_is_continuous: torch.Tensor | None=None):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = sample * 0.5
        if phase_is_continuous is None:
            shift_mask = (torch.rand([batch_size], device=device) < 0.5).float()
        else:
            if phase_is_continuous.shape != (batch_size,):
                raise ValueError(f'phase_is_continuous must have shape ({batch_size},), got {tuple(phase_is_continuous.shape)}.')
            shift_mask = (~phase_is_continuous.to(device=device, dtype=torch.bool)).to(dtype)
        sample = sample + shift_mask * 0.5
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _extract_action_token_count(self, continuous_features: torch.Tensor, discrete_features: torch.Tensor) -> int:
        if self.share_action_tokens_by_dim:
            return continuous_features.shape[1]
        return continuous_features.shape[1] + discrete_features.shape[1]

    def _build_action_features(self, continuous_actions: torch.Tensor, discrete_tokens: torch.Tensor, continuous_timesteps: torch.Tensor, discrete_timesteps: torch.Tensor):
        continuous_features = self.continuous_encoder(continuous_actions, continuous_timesteps)
        discrete_features = self.discrete_encoder(discrete_tokens, discrete_timesteps)
        if self.share_action_tokens_by_dim:
            action_features = self.shared_action_fusion(continuous_features, discrete_features)
        else:
            action_features = torch.cat([continuous_features, discrete_features], dim=1)
        return (continuous_features, discrete_features, action_features)

    def _build_sequence_embeddings(self, continuous_actions: torch.Tensor, discrete_tokens: torch.Tensor, state: torch.Tensor | None, global_timesteps: torch.Tensor, continuous_timesteps: torch.Tensor, discrete_timesteps: torch.Tensor):
        device = continuous_actions.device
        batch_size = continuous_actions.shape[0]
        continuous_features, discrete_features, action_features = self._build_action_features(continuous_actions=continuous_actions, discrete_tokens=discrete_tokens, continuous_timesteps=continuous_timesteps, discrete_timesteps=discrete_timesteps)
        state_features = self.state_encoder(state) if state is not None else None
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        temb = self.model.timestep_encoder(global_timesteps)
        continuous_temb = self.model.continuous_timestep_encoder(continuous_timesteps)
        discrete_temb = self.model.discrete_timestep_encoder(discrete_timesteps)
        state_len = state_features.shape[1] if state_features is not None else 0
        future_len = future_tokens.shape[1]
        temb_expanded = temb.unsqueeze(1).expand(-1, sa_embs.shape[1], -1)
        temb_parts = []
        if state_len > 0:
            temb_parts.append(temb_expanded[:, :state_len])
        temb_parts.append(temb_expanded[:, state_len:state_len + future_len])
        if self.share_action_tokens_by_dim:
            action_len = action_features.shape[1]
            phase_is_continuous = (global_timesteps >= self.num_timestep_buckets // 2).to(dtype=continuous_temb.dtype)
            phase_is_continuous = phase_is_continuous.unsqueeze(1).unsqueeze(2)
            continuous_temb_expanded = continuous_temb.unsqueeze(1).expand(-1, action_len, -1)
            discrete_temb_expanded = discrete_temb.unsqueeze(1).expand(-1, action_len, -1)
            shared_action_temb = phase_is_continuous * continuous_temb_expanded + (1.0 - phase_is_continuous) * discrete_temb_expanded
            temb_parts.append(shared_action_temb)
        else:
            cont_len = continuous_features.shape[1]
            disc_len = discrete_features.shape[1] if discrete_features is not None else 0
            continuous_temb_expanded = continuous_temb.unsqueeze(1).expand(-1, cont_len, -1)
            discrete_temb_expanded = discrete_temb.unsqueeze(1).expand(-1, disc_len, -1) if disc_len > 0 else None
            temb_parts.append(continuous_temb_expanded)
            if discrete_temb_expanded is not None:
                temb_parts.append(discrete_temb_expanded)
        temb = torch.cat(temb_parts, dim=1)
        return (sa_embs, temb, state_features, future_tokens)

    def _extract_pred_continuous_actions(self, pred_continuous: torch.Tensor) -> torch.Tensor:
        if self.share_action_tokens_by_dim:
            pred_continuous_tokens = pred_continuous[:, -self.discrete_horizon:, :]
            return pred_continuous_tokens.view(pred_continuous.shape[0], self.continuous_horizon, self.continuous_dim)
        if self.isolate_branch_tokens_before_shared:
            return pred_continuous[:, -self.continuous_horizon:]
        if self.branch_input_mode == 'full':
            return pred_continuous[:, -(self.continuous_horizon + self.discrete_horizon):-self.discrete_horizon]
        return pred_continuous[:, -self.continuous_horizon:]

    def _extract_pred_discrete_actions(self, pred_discrete: torch.Tensor) -> torch.Tensor:
        return pred_discrete[:, -self.discrete_horizon:]

    def _run_discrete_condition_rollout(self, vl_embs_list: list, continuous_actions: torch.Tensor, state: torch.Tensor=None, num_discrete_steps: int | None=None, discrete_rollout_mode: str | None=None, encoder_attention_mask: torch.Tensor=None) -> torch.Tensor:
        if num_discrete_steps is None:
            num_discrete_steps = self.default_num_discrete_steps
        if num_discrete_steps is None:
            num_discrete_steps = 3
        num_discrete_steps = int(num_discrete_steps)
        if num_discrete_steps <= 0:
            raise ValueError(f'continuous_condition_rollout requires num_discrete_steps > 0, got {num_discrete_steps}.')
        discrete_rollout_mode = self._validate_discrete_rollout_mode(self.discrete_rollout_mode if discrete_rollout_mode is None else discrete_rollout_mode)
        batch_size = continuous_actions.shape[0]
        device = continuous_actions.device
        discrete_actions = torch.full(size=(batch_size, self.discrete_horizon), fill_value=self.num_discrete_bins, dtype=torch.long, device=device)
        for step in range(num_discrete_steps):
            discrete_t_cont = step / float(num_discrete_steps)
            global_t_cont = 0.5 * discrete_t_cont
            global_step = min(int(global_t_cont * self.num_timestep_buckets), self.num_timestep_buckets - 1)
            continuous_step = 0
            discrete_step = min(int(discrete_t_cont * self.num_timestep_buckets), self.num_timestep_buckets - 1)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=global_step, device=device, dtype=torch.long)
            continuous_timesteps_tensor = torch.full(size=(batch_size,), fill_value=continuous_step, device=device, dtype=torch.long)
            discrete_timesteps_tensor = torch.full(size=(batch_size,), fill_value=discrete_step, device=device, dtype=torch.long)
            sa_embs, temb, _, _ = self._build_sequence_embeddings(continuous_actions=continuous_actions, discrete_tokens=discrete_actions, state=state, global_timesteps=timesteps_tensor, continuous_timesteps=continuous_timesteps_tensor, discrete_timesteps=discrete_timesteps_tensor)
            pred_discrete_actions = self._predict_discrete_logits_from_embeddings(vl_embs_list=vl_embs_list, sa_embs=sa_embs, temb=temb, encoder_attention_mask=encoder_attention_mask)
            probs = F.softmax(pred_discrete_actions, dim=-1)
            confidence, predicted_tokens = probs.max(dim=-1)
            if step == num_discrete_steps - 1:
                discrete_actions = predicted_tokens
                continue
            if discrete_rollout_mode == 'refresh_all':
                discrete_actions = predicted_tokens
            else:
                num_keep = int(self.discrete_horizon * ((step + 1) / float(num_discrete_steps)))
                if num_keep <= 0:
                    discrete_actions = torch.full_like(discrete_actions, self.num_discrete_bins)
                elif num_keep >= self.discrete_horizon:
                    discrete_actions = predicted_tokens
                else:
                    _, top_k_indices = confidence.topk(num_keep, dim=-1)
                    new_discrete_actions = torch.full_like(discrete_actions, self.num_discrete_bins)
                    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_keep)
                    new_discrete_actions[batch_indices, top_k_indices] = predicted_tokens[batch_indices, top_k_indices]
                    discrete_actions = new_discrete_actions
        return discrete_actions

    def _predict_discrete_logits_from_embeddings(self, vl_embs_list: list, sa_embs: torch.Tensor, temb: torch.Tensor, detach_shared_output: bool=False, encoder_attention_mask: torch.Tensor=None) -> torch.Tensor:
        sa_embs, temb = self._isolate_branch_sequence(sa_embs, temb, 'discrete')
        model_output = sa_embs
        if detach_shared_output:
            with torch.no_grad():
                model_output = self._run_shared_blocks(vl_embs_list=vl_embs_list, hidden_states=model_output, temb=temb, encoder_attention_mask=encoder_attention_mask, detach_encoder_hidden_states=True)
            model_output = model_output.detach()
            temb = temb.detach()
        else:
            model_output = self._run_shared_blocks(vl_embs_list=vl_embs_list, hidden_states=model_output, temb=temb, encoder_attention_mask=encoder_attention_mask)
        _, model_output = self._run_branch_blocks(vl_embs_list=vl_embs_list, shared_output=model_output, temb=temb, run_continuous=self.branch_communication_on_self_attention, run_discrete=True, detach_encoder_hidden_states=detach_shared_output, encoder_attention_mask=encoder_attention_mask)
        pred_discrete = self.discrete_decoder(model_output)
        pred_discrete_actions = self._extract_pred_discrete_actions(pred_discrete)
        return pred_discrete_actions

    def _is_branch_self_attention_layer(self, branch_layer_idx: int) -> bool:
        global_idx = len(self.model.shared_transformer_blocks) + branch_layer_idx
        return bool(getattr(self.model.config, 'interleave_self_attention', False)) and global_idx % 2 == 1

    def _run_shared_blocks(self, vl_embs_list: list, hidden_states: torch.Tensor, temb: torch.Tensor, encoder_attention_mask: torch.Tensor=None, detach_encoder_hidden_states: bool=False) -> torch.Tensor:
        interleave = bool(getattr(self.model.config, 'interleave_self_attention', False))
        for layer_idx, layer in enumerate(self.model.shared_transformer_blocks):
            is_self_attention_layer = interleave and layer_idx % 2 == 1
            encoder_hidden_states = None
            if not is_self_attention_layer:
                encoder_hidden_states = vl_embs_list[layer_idx]
                if detach_encoder_hidden_states:
                    encoder_hidden_states = encoder_hidden_states.detach()
            hidden_states = layer(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask if encoder_hidden_states is not None else None, temb=temb)
        return hidden_states

    @staticmethod
    def _slice_token_conditioning(temb: torch.Tensor | None, token_slice) -> torch.Tensor | None:
        if temb is None or temb.dim() != 3:
            return temb
        if isinstance(token_slice, tuple):
            return torch.cat([temb[:, part] for part in token_slice], dim=1)
        return temb[:, token_slice]

    def _isolate_branch_sequence(self, hidden_states: torch.Tensor, temb: torch.Tensor, branch: str) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.isolate_branch_tokens_before_shared:
            return (hidden_states, temb)
        future_len = int(self.config.num_target_vision_tokens)
        continuous_len = int(self.continuous_horizon)
        discrete_len = int(self.discrete_horizon)
        state_len = hidden_states.shape[1] - future_len - continuous_len - discrete_len
        if state_len < 0:
            raise ValueError('Cannot isolate branch sequence: input is shorter than future + continuous + discrete tokens.')
        if branch == 'continuous':
            token_parts = slice(0, state_len + future_len + continuous_len)
        elif branch == 'discrete':
            token_parts = (slice(0, state_len + future_len), slice(hidden_states.shape[1] - discrete_len, hidden_states.shape[1]))
        else:
            raise ValueError(f'Unknown branch {branch!r}.')
        return (self._slice_token_conditioning(hidden_states, token_parts), self._slice_token_conditioning(temb, token_parts))

    def _route_branch_inputs(self, shared_output: torch.Tensor, temb: torch.Tensor | None, run_continuous: bool, run_discrete: bool) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.branch_input_mode == 'full':
            return (shared_output if run_continuous else None, shared_output.clone() if run_discrete else None, temb if run_continuous else None, temb if run_discrete else None)
        if self.share_action_tokens_by_dim:
            raise ValueError(f'branch_input_mode={self.branch_input_mode!r} requires separate continuous/discrete action tokens.')
        future_len = int(self.config.num_target_vision_tokens)
        continuous_len = int(self.continuous_horizon)
        discrete_len = int(self.discrete_horizon)
        state_len = shared_output.shape[1] - future_len - continuous_len - discrete_len
        if state_len < 0:
            raise ValueError('Cannot route branch inputs: sequence is shorter than future + continuous + discrete tokens.')
        future_slice = slice(state_len, state_len + future_len)
        continuous_slice = slice(state_len + future_len, state_len + future_len + continuous_len)
        discrete_slice = slice(shared_output.shape[1] - discrete_len, shared_output.shape[1])
        if self.branch_input_mode == 'action_only':
            continuous_parts = continuous_slice
            discrete_parts = discrete_slice
        else:
            continuous_parts = slice(state_len, state_len + future_len + continuous_len)
            discrete_parts = (future_slice, discrete_slice)
        continuous_output = None
        continuous_temb = None
        if run_continuous:
            continuous_output = self._slice_token_conditioning(shared_output, continuous_parts)
            continuous_temb = self._slice_token_conditioning(temb, continuous_parts)
        discrete_output = None
        discrete_temb = None
        if run_discrete:
            discrete_output = self._slice_token_conditioning(shared_output, discrete_parts)
            discrete_temb = self._slice_token_conditioning(temb, discrete_parts)
        return (continuous_output, discrete_output, continuous_temb, discrete_temb)

    def _run_branch_blocks(self, vl_embs_list: list, shared_output: torch.Tensor, temb: torch.Tensor, run_continuous: bool=True, run_discrete: bool=True, detach_encoder_hidden_states: bool=False, enable_branch_communication: bool | None=None, encoder_attention_mask: torch.Tensor=None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if enable_branch_communication is None:
            enable_branch_communication = self.branch_communication_on_self_attention
        continuous_input, discrete_input, continuous_temb_input, discrete_temb_input = self._route_branch_inputs(shared_output=shared_output, temb=temb, run_continuous=run_continuous, run_discrete=run_discrete)
        continuous_output = self.model.continuous_branch_in_proj(continuous_input) if run_continuous else None
        discrete_output = self.model.discrete_branch_in_proj(discrete_input) if run_discrete else None
        continuous_temb = self.model.continuous_branch_temb_proj(continuous_temb_input) if run_continuous else None
        discrete_temb = self.model.discrete_branch_temb_proj(discrete_temb_input) if run_discrete else None
        num_shared_layers = len(self.model.shared_transformer_blocks)
        for layer_idx, (continuous_layer, discrete_layer) in enumerate(zip(self.model.continuous_transformer_blocks, self.model.discrete_transformer_blocks)):
            is_self_attention_layer = self._is_branch_self_attention_layer(layer_idx)
            if enable_branch_communication and is_self_attention_layer and (continuous_output is not None) and (discrete_output is not None):
                continuous_len = continuous_output.shape[1]
                joint_output = torch.cat([continuous_output, discrete_output], dim=1)
                joint_continuous_temb = continuous_temb
                joint_discrete_temb = discrete_temb
                if continuous_temb is not None and continuous_temb.dim() == 3:
                    joint_continuous_temb = torch.cat([continuous_temb, continuous_temb], dim=1)
                if discrete_temb is not None and discrete_temb.dim() == 3:
                    joint_discrete_temb = torch.cat([discrete_temb, discrete_temb], dim=1)
                continuous_joint = continuous_layer(hidden_states=joint_output, encoder_hidden_states=None, temb=joint_continuous_temb)
                discrete_joint = discrete_layer(hidden_states=joint_output, encoder_hidden_states=None, temb=joint_discrete_temb)
                continuous_output = continuous_joint[:, :continuous_len]
                discrete_output = discrete_joint[:, continuous_len:]
                continue
            encoder_hidden_states = None
            if not is_self_attention_layer:
                encoder_hidden_states = vl_embs_list[layer_idx + num_shared_layers]
                if detach_encoder_hidden_states:
                    encoder_hidden_states = encoder_hidden_states.detach()
            if continuous_output is not None:
                continuous_output = continuous_layer(hidden_states=continuous_output, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask if encoder_hidden_states is not None else None, temb=continuous_temb)
            if discrete_output is not None:
                discrete_output = discrete_layer(hidden_states=discrete_output, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask if encoder_hidden_states is not None else None, temb=discrete_temb)
        if continuous_output is not None:
            continuous_output = self.model.continuous_branch_out_proj(continuous_output)
        if discrete_output is not None:
            discrete_output = self.model.discrete_branch_out_proj(discrete_output)
        return (continuous_output, discrete_output)

    def _predict_continuous_velocity(self, vl_embs_list: list, continuous_actions: torch.Tensor, discrete_tokens: torch.Tensor, state: torch.Tensor, global_timesteps: torch.Tensor, continuous_timesteps: torch.Tensor, discrete_timesteps: torch.Tensor, detach_shared_output: bool=False, encoder_attention_mask: torch.Tensor=None) -> torch.Tensor:
        if detach_shared_output:
            with torch.no_grad():
                sa_embs, temb, _, _ = self._build_sequence_embeddings(continuous_actions=continuous_actions, discrete_tokens=discrete_tokens, state=state, global_timesteps=global_timesteps, continuous_timesteps=continuous_timesteps, discrete_timesteps=discrete_timesteps)
        else:
            sa_embs, temb, _, _ = self._build_sequence_embeddings(continuous_actions=continuous_actions, discrete_tokens=discrete_tokens, state=state, global_timesteps=global_timesteps, continuous_timesteps=continuous_timesteps, discrete_timesteps=discrete_timesteps)
        sa_embs, temb = self._isolate_branch_sequence(sa_embs, temb, 'continuous')
        model_output = sa_embs
        if detach_shared_output:
            with torch.no_grad():
                model_output = self._run_shared_blocks(vl_embs_list=vl_embs_list, hidden_states=model_output, temb=temb, encoder_attention_mask=encoder_attention_mask, detach_encoder_hidden_states=True)
            model_output = model_output.detach()
            temb = temb.detach()
        else:
            model_output = self._run_shared_blocks(vl_embs_list=vl_embs_list, hidden_states=model_output, temb=temb, encoder_attention_mask=encoder_attention_mask)
        model_output, _ = self._run_branch_blocks(vl_embs_list=vl_embs_list, shared_output=model_output, temb=temb, run_continuous=True, run_discrete=self.branch_communication_on_self_attention, detach_encoder_hidden_states=detach_shared_output, encoder_attention_mask=encoder_attention_mask)
        pred_continuous = self.continuous_decoder(model_output)
        return self._extract_pred_continuous_actions(pred_continuous)

    def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor=None, encoder_attention_mask: torch.Tensor=None, phase_is_continuous: torch.Tensor=None, discrete_actions: torch.Tensor=None):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, D_action)
        """
        continuous_actions = actions.clone()
        if discrete_actions is None:
            discrete_actions = self.discretize_actions(actions)
        expected_discrete_shape = (actions.shape[0], self.discrete_horizon)
        if discrete_actions.shape != expected_discrete_shape:
            raise ValueError(f'Expected discrete actions shaped {expected_discrete_shape}, got {tuple(discrete_actions.shape)}.')
        device = continuous_actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        noise = torch.randn(continuous_actions.shape, device=continuous_actions.device, dtype=continuous_actions.dtype)
        t = self.sample_time(continuous_actions.shape[0], device=continuous_actions.device, dtype=continuous_actions.dtype, phase_is_continuous=phase_is_continuous)
        t = t[:, None, None]
        if phase_is_continuous is None:
            mask_low = (t <= 0.5).float()
        else:
            phase_is_continuous = phase_is_continuous.to(device=device, dtype=torch.bool)
            phase_mask = phase_is_continuous[:, None, None]
            t = torch.where(phase_mask, t.clamp(min=0.5, max=1.0), t.clamp(min=0.0, max=0.5))
            mask_low = (~phase_mask).to(dtype=t.dtype)
        continuous_t = mask_low * 0.0 + (1 - mask_low) * ((t - 0.5) * 2)
        discrete_t = mask_low * (t * 2) + (1 - mask_low) * 1.0
        t_discretized = torch.clamp((t[:, 0, 0] * self.num_timestep_buckets).long(), max=self.num_timestep_buckets - 1)
        continuous_t_discretized = torch.clamp((continuous_t[:, 0, 0] * self.num_timestep_buckets).long(), max=self.num_timestep_buckets - 1)
        discrete_t_discretized = torch.clamp((discrete_t[:, 0, 0] * self.num_timestep_buckets).long(), max=self.num_timestep_buckets - 1)
        d_L = discrete_actions.shape[1]
        continuous_half_mask = mask_low[:, 0, 0] == 0
        discrete_half_mask = ~continuous_half_mask
        noise_ratio = 1.0 - discrete_t
        noisy_discrete = discrete_actions.clone()
        for b in range(B):
            if self.continuous_condition_use_source_alpha and continuous_half_mask[b]:
                num_noise = int(round(d_L * (1.0 - self.continuous_source_alpha)))
            else:
                num_noise = int(d_L * noise_ratio[b].item())
            if num_noise > 0:
                noise_pos = torch.randperm(d_L, device=device)[:num_noise]
                if self.continuous_condition_use_source_alpha and continuous_half_mask[b]:
                    noisy_discrete[b, noise_pos] = self._sample_other_bins(discrete_actions[b, noise_pos], self.num_discrete_bins)
                else:
                    noisy_discrete[b, noise_pos] = self.num_discrete_bins
        noisy_discrete = noisy_discrete.clamp(min=0, max=self.num_discrete_bins)
        discrete_targets = discrete_actions.long()
        original_mask_positions = noisy_discrete == self.num_discrete_bins
        # Keep these RNG draws so the sampling sequence remains unchanged.
        reserved_selection_1 = continuous_half_mask & (torch.rand(B, device=device) < 0.0)
        reserved_selection_2 = continuous_half_mask & ~reserved_selection_1 & (torch.rand(B, device=device) < 0.0)
        source_discrete_tokens = discrete_actions
        mixed_source_actions = self._build_continuous_source_actions(noise_actions=noise, discrete_tokens=source_discrete_tokens)
        if self.continuous_source_continuous_phase_only:
            source_actions = torch.where(continuous_half_mask[:, None, None], mixed_source_actions, noise)
        else:
            source_actions = mixed_source_actions
        noisy_trajectory = (1 - continuous_t) * source_actions + continuous_t * continuous_actions
        velocity = continuous_actions - source_actions
        one_step_loss = torch.zeros(B, device=device, dtype=torch.float32)
        if self.continuous_one_step_loss_weight > 0.0 and self.continuous_one_step_loss_prob > 0.0 and bool(continuous_half_mask.any()):
            one_step_selected = continuous_half_mask & (torch.rand(B, device=device) < self.continuous_one_step_loss_prob)
            if bool(one_step_selected.any()):
                one_step_indices = torch.where(one_step_selected)[0]
                one_step_vl_embs_list = [vl_embs[one_step_indices] for vl_embs in vl_embs_list]
                one_step_state = state[one_step_indices] if state is not None else None
                one_step_attention_mask = encoder_attention_mask[one_step_indices] if encoder_attention_mask is not None else None
                one_step_batch = one_step_indices.shape[0]
                one_step_global_timesteps = torch.full((one_step_batch,), fill_value=min(self.num_timestep_buckets // 2, self.num_timestep_buckets - 1), device=device, dtype=torch.long)
                one_step_continuous_timesteps = torch.zeros((one_step_batch,), device=device, dtype=torch.long)
                one_step_discrete_timesteps = torch.full((one_step_batch,), fill_value=self.num_timestep_buckets - 1, device=device, dtype=torch.long)
                one_step_velocity = self._predict_continuous_velocity(vl_embs_list=one_step_vl_embs_list, continuous_actions=source_actions[one_step_indices], discrete_tokens=noisy_discrete[one_step_indices], state=one_step_state, global_timesteps=one_step_global_timesteps, continuous_timesteps=one_step_continuous_timesteps, discrete_timesteps=one_step_discrete_timesteps, detach_shared_output=False, encoder_attention_mask=one_step_attention_mask)
                one_step_actions = source_actions[one_step_indices] + one_step_velocity
                one_step_error = ((one_step_actions.float() - continuous_actions[one_step_indices].float()) ** 2).mean(dim=[1, 2])
                one_step_loss[one_step_indices] = one_step_error
        discrete_one_step_loss = torch.zeros(B, device=device, dtype=torch.float32)
        reserved_selection_3 = discrete_half_mask & (torch.rand(B, device=device) < 0.0)
        sa_embs, temb, _, _ = self._build_sequence_embeddings(continuous_actions=noisy_trajectory, discrete_tokens=noisy_discrete, state=state, global_timesteps=t_discretized, continuous_timesteps=continuous_t_discretized, discrete_timesteps=discrete_t_discretized)
        use_disjoint_routing = self.disjoint_branch_batch_routing
        model_output = None
        if not (self.isolate_branch_tokens_before_shared and use_disjoint_routing):
            model_output = self._run_shared_blocks(vl_embs_list=vl_embs_list, hidden_states=sa_embs, temb=temb, encoder_attention_mask=encoder_attention_mask)
        if use_disjoint_routing:
            if phase_is_continuous is None:
                raise ValueError('disjoint_branch_batch_routing requires paired phase assignments from the framework.')
            continuous_indices = torch.where(continuous_half_mask)[0]
            discrete_indices = torch.where(discrete_half_mask)[0]
            if continuous_indices.numel() == 0 or discrete_indices.numel() == 0:
                raise RuntimeError('Disjoint branch routing requires non-empty continuous and discrete subsets on every rank.')
            continuous_vl_embs = [hidden[continuous_indices] for hidden in vl_embs_list]
            continuous_attention_mask = encoder_attention_mask[continuous_indices] if encoder_attention_mask is not None else None
            continuous_temb = temb[continuous_indices]
            if self.isolate_branch_tokens_before_shared:
                continuous_input, continuous_temb = self._isolate_branch_sequence(sa_embs[continuous_indices], continuous_temb, 'continuous')
                continuous_shared_output = self._run_shared_blocks(vl_embs_list=continuous_vl_embs, hidden_states=continuous_input, temb=continuous_temb, encoder_attention_mask=continuous_attention_mask)
            else:
                continuous_shared_output = model_output[continuous_indices]
            continuous_output, _ = self._run_branch_blocks(vl_embs_list=continuous_vl_embs, shared_output=continuous_shared_output, temb=continuous_temb, run_continuous=True, run_discrete=False, encoder_attention_mask=continuous_attention_mask)
            discrete_vl_embs = [hidden[discrete_indices] for hidden in vl_embs_list]
            discrete_attention_mask = encoder_attention_mask[discrete_indices] if encoder_attention_mask is not None else None
            discrete_temb = temb[discrete_indices]
            if self.isolate_branch_tokens_before_shared:
                discrete_input, discrete_temb = self._isolate_branch_sequence(sa_embs[discrete_indices], discrete_temb, 'discrete')
                discrete_shared_output = self._run_shared_blocks(vl_embs_list=discrete_vl_embs, hidden_states=discrete_input, temb=discrete_temb, encoder_attention_mask=discrete_attention_mask)
            else:
                discrete_shared_output = model_output[discrete_indices]
            _, discrete_output = self._run_branch_blocks(vl_embs_list=discrete_vl_embs, shared_output=discrete_shared_output, temb=discrete_temb, run_continuous=False, run_discrete=True, encoder_attention_mask=discrete_attention_mask)
            pred_continuous = self.continuous_decoder(continuous_output)
            pred_continuous_actions = self._extract_pred_continuous_actions(pred_continuous)
            pred_discrete = self.discrete_decoder(discrete_output)
            pred_discrete_actions = self._extract_pred_discrete_actions(pred_discrete)
        else:
            model_output, discretize_model_output = self._run_branch_blocks(vl_embs_list=vl_embs_list, shared_output=model_output, temb=temb, run_continuous=True, run_discrete=True, encoder_attention_mask=encoder_attention_mask)
            pred_continuous = self.continuous_decoder(model_output)
            pred_continuous_actions = self._extract_pred_continuous_actions(pred_continuous)
            pred_discrete = self.discrete_decoder(discretize_model_output)
            pred_discrete_actions = self._extract_pred_discrete_actions(pred_discrete)
        continuous_targets = velocity[continuous_indices] if use_disjoint_routing else velocity
        continuous_loss = ((pred_continuous_actions - continuous_targets) ** 2).mean(dim=[1, 2])
        if self.continuous_one_step_loss_weight > 0.0:
            selected_one_step_loss = one_step_loss[continuous_indices] if use_disjoint_routing else one_step_loss
            continuous_loss = continuous_loss + self.continuous_one_step_loss_weight * selected_one_step_loss
        selected_discrete_targets = discrete_targets[discrete_indices] if use_disjoint_routing else discrete_targets
        discrete_loss_full = F.cross_entropy(pred_discrete_actions.reshape(-1, self.num_discrete_bins), selected_discrete_targets.reshape(-1), reduction='none').view(selected_discrete_targets.shape[0], -1)
        mask_positions = (original_mask_positions[discrete_indices] if use_disjoint_routing else original_mask_positions).float()
        visible_positions = 1.0 - mask_positions
        mask_counts = mask_positions.sum(dim=1).clamp(min=1.0)
        visible_counts = visible_positions.sum(dim=1).clamp(min=1.0)
        masked_discrete_loss = (discrete_loss_full * mask_positions).sum(dim=1) / mask_counts
        visible_discrete_loss = (discrete_loss_full * visible_positions).sum(dim=1) / visible_counts
        if self.discrete_visible_loss_weight > 0.0:
            discrete_loss = masked_discrete_loss + self.discrete_visible_loss_weight * visible_discrete_loss
        elif bool(getattr(self.config, 'discrete_loss_on_all_tokens', False)):
            discrete_loss = discrete_loss_full.mean(dim=1)
        else:
            discrete_loss = masked_discrete_loss
        if use_disjoint_routing:
            reduced_continuous_loss = continuous_loss.mean()
            reduced_discrete_loss = discrete_loss.mean()
        elif bool(getattr(self.config, 'joint_branch_supervision', False)):
            reduced_continuous_loss = continuous_loss.mean()
            reduced_discrete_loss = discrete_loss.mean()
        else:
            mask_low = mask_low.squeeze()
            continuous_mask = 1.0 - mask_low
            discrete_mask = mask_low
            reduced_continuous_loss = (continuous_loss * continuous_mask).sum() / continuous_mask.sum().clamp(min=1)
            reduced_discrete_loss = (discrete_loss * discrete_mask).sum() / discrete_mask.sum().clamp(min=1)
        return (reduced_continuous_loss, reduced_discrete_loss)

    @torch.no_grad()
    def predict_action(self, vl_embs_list: list, state: torch.Tensor=None, num_inference_timesteps: int=None, num_discrete_steps: int=None, num_continuous_steps: int=None, generator: torch.Generator=None, discrete_rollout_mode: str=None, encoder_attention_mask: torch.Tensor=None) -> torch.Tensor:
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        discrete_rollout_mode = self._validate_discrete_rollout_mode(self.discrete_rollout_mode if discrete_rollout_mode is None else discrete_rollout_mode)
        inference_branch_mode = 'hybrid'
        if generator is None:
            continuous_actions = torch.randn(size=(batch_size, self.continuous_horizon, self.continuous_dim), dtype=vl_embs_list[0].dtype, device=device)
        else:
            continuous_actions = torch.randn(size=(batch_size, self.continuous_horizon, self.continuous_dim), dtype=torch.float32, generator=generator).to(device=device, dtype=vl_embs_list[0].dtype)
        discrete_actions = torch.full(size=(batch_size, self.discrete_horizon), fill_value=self.num_discrete_bins, dtype=torch.long, device=device)
        total_steps = max(int(self.num_inference_timesteps), 2)
        num_discrete_steps = self.default_num_discrete_steps if num_discrete_steps is None else int(num_discrete_steps)
        num_continuous_steps = self.default_num_continuous_steps if num_continuous_steps is None else int(num_continuous_steps)
        continuous_dt = 1.0 / num_continuous_steps if num_continuous_steps else 0.0

        def build_condition_inputs(global_t_cont: float, continuous_t_cont: float, discrete_t_cont: float):
            global_step = min(int(global_t_cont * self.num_timestep_buckets), self.num_timestep_buckets - 1)
            continuous_step = min(int(continuous_t_cont * self.num_timestep_buckets), self.num_timestep_buckets - 1)
            discrete_step = min(int(discrete_t_cont * self.num_timestep_buckets), self.num_timestep_buckets - 1)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=global_step, device=device, dtype=torch.long)
            continuous_timesteps_tensor = torch.full(size=(batch_size,), fill_value=continuous_step, device=device, dtype=torch.long)
            discrete_timesteps_tensor = torch.full(size=(batch_size,), fill_value=discrete_step, device=device, dtype=torch.long)
            sa_embs, temb, _, _ = self._build_sequence_embeddings(continuous_actions=continuous_actions, discrete_tokens=discrete_actions, state=state, global_timesteps=timesteps_tensor, continuous_timesteps=continuous_timesteps_tensor, discrete_timesteps=discrete_timesteps_tensor)
            return (sa_embs, temb)
        discrete_actions = self._run_discrete_condition_rollout(vl_embs_list=vl_embs_list, continuous_actions=continuous_actions, state=state, num_discrete_steps=num_discrete_steps, discrete_rollout_mode=discrete_rollout_mode, encoder_attention_mask=encoder_attention_mask)
        continuous_actions = self._build_continuous_source_actions(noise_actions=continuous_actions, discrete_tokens=discrete_actions)
        for step in range(num_continuous_steps):
            continuous_t_cont = step / float(num_continuous_steps)
            global_t_cont = 0.5 + 0.5 * continuous_t_cont
            sa_embs, temb = build_condition_inputs(global_t_cont=global_t_cont, continuous_t_cont=continuous_t_cont, discrete_t_cont=1.0)
            sa_embs, temb = self._isolate_branch_sequence(sa_embs, temb, 'continuous')
            model_output = self._run_shared_blocks(vl_embs_list=vl_embs_list, hidden_states=sa_embs, temb=temb, encoder_attention_mask=encoder_attention_mask)
            model_output, _ = self._run_branch_blocks(vl_embs_list=vl_embs_list, shared_output=model_output, temb=temb, run_continuous=True, run_discrete=self.branch_communication_on_self_attention, encoder_attention_mask=encoder_attention_mask)
            pred_continuous = self.continuous_decoder(model_output)
            pred_velocity = self._extract_pred_continuous_actions(pred_continuous)
            continuous_actions = continuous_actions + continuous_dt * pred_velocity
        discrete_actions_continuous = self._decode_discrete_tokens_to_actions(discrete_actions)
        discrete_actions_coarse = self.decoded_discrete_action(discrete_actions).view(batch_size, self.discrete_action_horizon, self.continuous_dim)
        return {'continuous_actions': continuous_actions, 'discrete_actions_continuous': discrete_actions_continuous, 'discrete_actions_coarse': discrete_actions_coarse, 'discrete_actions_tokens': discrete_actions, 'num_discrete_steps': num_discrete_steps, 'num_continuous_steps': num_continuous_steps, 'discrete_rollout_mode': discrete_rollout_mode, 'inference_branch_mode': 'hybrid'}

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return HybridLayerwiseFlowmatchingActionHead(global_config=config)
