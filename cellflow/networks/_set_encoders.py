from collections.abc import Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from flax.typing import FrozenDict

from cellflow._types import ArrayLike, Layers_separate_input_t, Layers_t
from cellflow.networks import _utils as nn_utils

__all__ = [
    "ConditionEncoder",
]


class SlotwiseConditionFusion(nn.Module):
    """Fuse linked perturbation covariates on each set element before pooling."""

    linked_groups: Sequence[str]
    mlp_dims: Sequence[int] = (128, 128)
    modulation_scale: float = 0.5
    token_dim: int | None = None
    layer_norm: bool = True
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        primary_tokens: jnp.ndarray,
        linked_tokens: dict[str, jnp.ndarray],
        training: bool = True,
    ) -> jnp.ndarray:
        token_dim = self.token_dim or primary_tokens.shape[-1]

        if self.token_dim is not None:
            fused_tokens = nn.Dense(self.token_dim, name="primary_proj")(primary_tokens)
        else:
            fused_tokens = primary_tokens

        for linked_group in self.linked_groups:
            linked = linked_tokens[linked_group]
            if self.mlp_dims:
                linked_hidden = nn_utils.MLPBlock(
                    dims=self.mlp_dims,
                    act_last_layer=True,
                    dropout_rate=self.dropout_rate,
                    name=f"{linked_group}_mlp",
                )(linked, training=training)
            else:
                linked_hidden = linked

            scale = nn.Dense(token_dim, name=f"{linked_group}_scale")(linked_hidden)
            shift = nn.Dense(token_dim, name=f"{linked_group}_shift")(linked_hidden)
            linked_strength = (
                linked
                if linked.shape[-1] == 1
                else jnp.linalg.norm(linked, axis=-1, keepdims=True)
            )
            fused_tokens = fused_tokens * (
                1.0 + self.modulation_scale * nn.tanh(scale)
            )
            fused_tokens = fused_tokens + linked_strength * shift

        if self.layer_norm:
            fused_tokens = nn.LayerNorm(name="slot_fusion_layer_norm")(fused_tokens)
        return fused_tokens


class ConditionEncoder(nn_utils.BaseModule):
    """
    Encoder for conditions represented as sets of perturbations.

    Parameters
    ----------
    output_dim
        Dimensionality of the output.
    condition_mode
        Mode of the encoder, should be one of:

        - ``'deterministic'``: Learns condition encoding point-wise.
        - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.
    regularization
        Regularization strength in the latent space:

        - For deterministic mode, it is the strength of the L2 regularization.
        - For stochastic mode, it is the strength of the KL divergence regularization.
    decoder
        Whether to use a decoder.
    pooling
        Pooling method, should be one of:

        - ``'mean'``: Aggregates combinations of covariates by the mean of their learned
          embeddings.
        - ``'attention_token'``: Aggregates combinations of covariates by an attention mechanism
          with a token.
        - ``'attention_seed'``: Aggregates combinations of covariates by an attention mechanism
          with a seed.
    pooling_kwargs
        Keyword arguments for the pooling method.
    covariates_not_pooled
        Covariates that will escape pooling (should be identical across all set elements).
    layers_before_pool
        Layers before pooling. Either a sequence of tuples with layer type and parameters or a
        dictionary with input-specific layers.
    layers_after_pool
        Layers after pooling.
    layers_decoder
        Layers for the decoder. Only relevant if ``'decoder'=True``.
    mask_value
        Value for masked elements used in input conditions.
    """

    output_dim: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 0.0
    decoder: bool = False
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=list)
    layers_before_pool: Layers_t | Layers_separate_input_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    layers_decoder: Layers_t = dc_field(default_factory=lambda: [])
    output_dropout: float = 0.0
    mask_value: float = 0.0
    slot_fusion: dict[str, Any] | None = None

    def setup(self):
        """Initialize the modules."""
        self.slot_fusion_cfg = dict(self.slot_fusion or {})
        self.slot_fusion_enabled = bool(self.slot_fusion_cfg)
        if self.slot_fusion_enabled:
            primary_group = self.slot_fusion_cfg.get("primary_group")
            if not isinstance(primary_group, str):
                raise ValueError("`slot_fusion['primary_group']` must be provided as a string.")
            linked_groups = self.slot_fusion_cfg.get("linked_groups")
            if linked_groups is None:
                linked_group = self.slot_fusion_cfg.get("linked_group")
                linked_groups = [linked_group] if linked_group is not None else []
            if isinstance(linked_groups, str) or not isinstance(linked_groups, Sequence) or len(linked_groups) == 0:
                raise ValueError(
                    "`slot_fusion['linked_groups']` must contain at least one linked perturbation group."
                )
            self.slot_fusion_primary_group = primary_group
            self.slot_fusion_linked_groups = tuple(str(group) for group in linked_groups)
            self.slot_fusion_module = SlotwiseConditionFusion(
                linked_groups=self.slot_fusion_linked_groups,
                mlp_dims=tuple(self.slot_fusion_cfg.get("mlp_dims", (128, 128))),
                modulation_scale=float(self.slot_fusion_cfg.get("modulation_scale", 0.5)),
                token_dim=self.slot_fusion_cfg.get("token_dim"),
                layer_norm=bool(self.slot_fusion_cfg.get("layer_norm", True)),
                dropout_rate=float(self.slot_fusion_cfg.get("dropout_rate", 0.0)),
            )

        # modules before pooling
        self.separate_inputs = isinstance(self.layers_before_pool, (dict | FrozenDict))
        if self.separate_inputs:
            # different layers for different inputs, before_pool_modules is of type Layers_separate_input_t
            self.before_pool_modules: dict[str, list[nn.Module]] | list[nn.Module] = {
                key: nn_utils._get_layers(layers)
                for key, layers in self.layers_before_pool.items()  # type: ignore[union-attr]
            }
        else:
            self.before_pool_modules = nn_utils._get_layers(self.layers_before_pool)  # type: ignore[arg-type]

        # pooling
        if self.pooling == "mean":
            self.pool_module = lambda x, mask, training: jnp.mean(x * mask, axis=-2)
        elif self.pooling == "attention_token":
            self.pool_module = nn_utils.TokenAttentionPooling(**self.pooling_kwargs)
        elif self.pooling == "attention_seed":
            self.pool_module = nn_utils.SeedAttentionPooling(**self.pooling_kwargs)

        # modules after pooling
        self.after_pool_modules_mean = nn_utils._get_layers(self.layers_after_pool, self.output_dim)

        if self.condition_mode == "stochastic":
            self.after_pool_modules_var = nn_utils._get_layers(self.layers_after_pool, self.output_dim)

    def __call__(
        self,
        conditions: dict[str, jnp.ndarray],
        training: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply the set encoder.

        Parameters
        ----------
        conditions : dict[str, jnp.ndarray]
            Dictionary of batch of conditions of shape ``(batch_size, set_size, condition_dim)``.
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        Mean and log-variance of conditions of shape ``(batch_size, output_dim)``.
        """
        mask, attention_mask = self._get_masks(conditions)
        conditions = self._apply_slot_fusion(conditions, training)

        # apply modules before pooling
        if self.separate_inputs:
            processed_inputs_pooling = []
            processed_inputs_other = []
            for pert_cov, conditions_i in conditions.items():
                # apply separate modules for all inputs
                conditions_i = nn_utils._apply_modules(
                    self.before_pool_modules[pert_cov],  # type: ignore[call-overload]
                    conditions_i,
                    attention_mask,
                    training,
                )
                if pert_cov in self.covariates_not_pooled:
                    # only keep first set element for covariates that are not pooled
                    processed_inputs_other.append(conditions_i[:, 0, :])
                else:
                    processed_inputs_pooling.append(conditions_i)

            conditions_pooling_arr = jnp.concatenate(processed_inputs_pooling, axis=-1)
            conditions_not_pooled = (
                jnp.concatenate(processed_inputs_other, axis=-1) if self.covariates_not_pooled else None
            )
        else:
            # by default, no modules before pooling for covariates that are not pooled
            if self.covariates_not_pooled:
                # divide conditions into pooled and not pooled
                conditions_not_pooled = []
                conditions_pooling = []
                for pert_cov in conditions:
                    if pert_cov in self.covariates_not_pooled:
                        conditions_not_pooled.append(conditions[pert_cov][:, 0, :])
                    else:
                        conditions_pooling.append(conditions[pert_cov])
                conditions_not_pooled = jnp.concatenate(
                    conditions_not_pooled,
                    axis=-1,
                )
                conditions_pooling_arr = jnp.concatenate(
                    conditions_pooling,
                    axis=-1,
                )

                # apply modules to pooled covariates
                conditions_pooling_arr = nn_utils._apply_modules(
                    self.before_pool_modules,  # type: ignore[arg-type]
                    conditions_pooling_arr,
                    attention_mask,
                    training,
                )
            else:
                conditions = jnp.concatenate(list(conditions.values()), axis=-1)
                conditions_pooling_arr = nn_utils._apply_modules(
                    self.before_pool_modules,
                    conditions,
                    attention_mask,
                    training,  # type: ignore[arg-type]
                )

        # pooling
        pool_mask = mask if self.pooling == "mean" else attention_mask
        conditions = self.pool_module(conditions_pooling_arr, pool_mask, training=training)
        if self.covariates_not_pooled:
            conditions = jnp.concatenate([conditions, conditions_not_pooled], axis=-1)

        # apply modules after pooling
        conditions = nn_utils._apply_modules(self.after_pool_modules_mean, conditions, None, training)

        if self.condition_mode == "stochastic":
            conditions_logvar = nn_utils._apply_modules(self.after_pool_modules_var, conditions, None, training)
        else:
            conditions_logvar = jnp.zeros_like(conditions)
        return conditions, conditions_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        conditions: dict[str, jnp.ndarray],
        **kwargs: Any,
    ):
        """Create initial training state."""
        params = self.init(
            rng,
            conditions={k: jnp.empty((1, v.shape[1], v.shape[2])) for k, v in conditions.items()},
            training=False,
        )["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            **kwargs,
        )

    def _get_masks(self, conditions: dict[str, ArrayLike]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get mask for padded conditions tensor."""
        # mask of shape (batch_size, set_size)
        mask = 1 - jnp.all(
            jnp.array(
                [jnp.all(c == self.mask_value, axis=-1) for c in conditions.values()],
            ),
            axis=0,
        )
        mask = jnp.expand_dims(mask, -1)

        # attention mask of shape (batch_size, 1, set_size, set_size)
        attention_mask = mask & jnp.matrix_transpose(mask)
        attention_mask = jnp.expand_dims(attention_mask, 1)

        return mask, attention_mask

    def _apply_slot_fusion(
        self,
        conditions: dict[str, jnp.ndarray],
        training: bool,
    ) -> dict[str, jnp.ndarray]:
        """Fuse linked perturbation covariates into the primary token representation."""
        if not self.slot_fusion_enabled:
            return conditions

        if self.slot_fusion_primary_group not in conditions:
            raise ValueError(
                f"Primary slot-fusion group '{self.slot_fusion_primary_group}' not found in condition inputs."
            )
        missing_groups = [
            group for group in self.slot_fusion_linked_groups if group not in conditions
        ]
        if missing_groups:
            raise ValueError(
                f"Linked slot-fusion groups missing from condition inputs: {missing_groups}."
            )

        fused_conditions = dict(conditions)
        fused_conditions[self.slot_fusion_primary_group] = self.slot_fusion_module(
            primary_tokens=conditions[self.slot_fusion_primary_group],
            linked_tokens={group: conditions[group] for group in self.slot_fusion_linked_groups},
            training=training,
        )
        for group in self.slot_fusion_linked_groups:
            fused_conditions.pop(group, None)
        return fused_conditions
