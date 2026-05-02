"""
Flow Matching for scBIG.

- ConditionalVelocityField: Neural velocity field with condition encoder
- ConditionEncoder: Set encoder for perturbation conditions
- OTFlowMatching: Optimal transport flow matching solver
"""

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import frozen_dict
from flax.training import train_state
from flax.typing import FrozenDict
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from scbig.models.networks import (
    MLPBlock, FilmBlock, ResNetBlock, 
    SelfAttentionBlock, TokenAttentionPooling, SeedAttentionPooling,
    sinusoidal_time_encoder
)


# ============================================================================
# Condition Encoder
# ============================================================================

class ConditionEncoder(nn.Module):
    """Encoder for perturbation conditions (set of perturbations)."""
    output_dim: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 0.0
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=list)
    layers_before_pool: list = dc_field(default_factory=lambda: [])
    layers_after_pool: list = dc_field(default_factory=lambda: [])
    output_dropout: float = 0.0
    mask_value: float = 0.0

    def setup(self):
        self.separate_inputs = isinstance(self.layers_before_pool, (dict, FrozenDict))
        
        if self.separate_inputs:
            self.before_pool_modules = {
                key: self._get_layers(layers)
                for key, layers in self.layers_before_pool.items()
            }
        else:
            self.before_pool_modules = self._get_layers(self.layers_before_pool)

        if self.pooling == "mean":
            self.pool_module = lambda x, mask, training: jnp.mean(x * mask, axis=-2)
        elif self.pooling == "attention_token":
            self.pool_module = TokenAttentionPooling(**self.pooling_kwargs)
        elif self.pooling == "attention_seed":
            self.pool_module = SeedAttentionPooling(**self.pooling_kwargs)

        self.after_pool_modules_mean = self._get_layers(self.layers_after_pool, self.output_dim)
        if self.condition_mode == "stochastic":
            self.after_pool_modules_var = self._get_layers(self.layers_after_pool, self.output_dim)

    def _get_layers(self, layers, output_dim=None):
        modules = []
        if isinstance(layers, Sequence):
            for layer in layers:
                layer = dict(layer)
                layer_type = layer.pop("layer_type", "mlp")
                if layer_type == "mlp":
                    modules.append(MLPBlock(**layer))
                elif layer_type == "self_attention":
                    modules.append(SelfAttentionBlock(**layer))
        if output_dim is not None:
            modules.append(nn.Dense(output_dim))
        return modules

    def _apply_modules(self, modules, conditions, attention_mask, training):
        for module in modules:
            if isinstance(module, SelfAttentionBlock):
                conditions = module(conditions, attention_mask, training)
            elif isinstance(module, nn.Dense):
                conditions = module(conditions)
            elif isinstance(module, nn.Dropout):
                conditions = module(conditions, deterministic=not training)
            else:
                conditions = module(conditions, training)
        return conditions

    def __call__(self, conditions: dict[str, jnp.ndarray], training: bool = True):
        mask, attention_mask = self._get_masks(conditions)

        if self.separate_inputs:
            processed_inputs_pooling = []
            processed_inputs_other = []
            for pert_cov, conditions_i in conditions.items():
                conditions_i = self._apply_modules(
                    self.before_pool_modules[pert_cov],
                    conditions_i, attention_mask, training
                )
                if pert_cov in self.covariates_not_pooled:
                    processed_inputs_other.append(conditions_i[:, 0, :])
                else:
                    processed_inputs_pooling.append(conditions_i)

            conditions_pooling_arr = jnp.concatenate(processed_inputs_pooling, axis=-1)
            conditions_not_pooled = (
                jnp.concatenate(processed_inputs_other, axis=-1) if self.covariates_not_pooled else None
            )
        else:
            if self.covariates_not_pooled:
                conditions_not_pooled = []
                conditions_pooling = []
                for pert_cov in conditions:
                    if pert_cov in self.covariates_not_pooled:
                        conditions_not_pooled.append(conditions[pert_cov][:, 0, :])
                    else:
                        conditions_pooling.append(conditions[pert_cov])
                conditions_not_pooled = jnp.concatenate(conditions_not_pooled, axis=-1)
                conditions_pooling_arr = jnp.concatenate(conditions_pooling, axis=-1)
                conditions_pooling_arr = self._apply_modules(
                    self.before_pool_modules, conditions_pooling_arr, attention_mask, training
                )
            else:
                conditions = jnp.concatenate(list(conditions.values()), axis=-1)
                conditions_pooling_arr = self._apply_modules(
                    self.before_pool_modules, conditions, attention_mask, training
                )

        pool_mask = mask if self.pooling == "mean" else attention_mask
        conditions = self.pool_module(conditions_pooling_arr, pool_mask, training=training)
        if self.covariates_not_pooled:
            conditions = jnp.concatenate([conditions, conditions_not_pooled], axis=-1)

        conditions = self._apply_modules(self.after_pool_modules_mean, conditions, None, training)

        if self.condition_mode == "stochastic":
            conditions_logvar = self._apply_modules(self.after_pool_modules_var, conditions, None, training)
        else:
            conditions_logvar = jnp.zeros_like(conditions)
        return conditions, conditions_logvar

    def _get_masks(self, conditions):
        mask = 1 - jnp.all(
            jnp.array([jnp.all(c == self.mask_value, axis=-1) for c in conditions.values()]),
            axis=0,
        )
        mask = jnp.expand_dims(mask, -1)
        attention_mask = mask & jnp.matrix_transpose(mask)
        attention_mask = jnp.expand_dims(attention_mask, 1)
        return mask, attention_mask


# ============================================================================
# Conditional Velocity Field
# ============================================================================

class ConditionalVelocityField(nn.Module):
    """Neural velocity field with condition encoder for flow matching."""
    output_dim: int
    max_combination_length: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 1.0
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: list = dc_field(default_factory=lambda: [])
    layers_after_pool: list = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_max_period: int = 10000
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    conditioning: Literal["concatenation", "film", "resnet"] = "concatenation"
    conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    def setup(self):
        if isinstance(self.conditioning_kwargs, dataclasses.Field):
            conditioning_kwargs = dict(self.conditioning_kwargs.default_factory())
        else:
            conditioning_kwargs = dict(self.conditioning_kwargs)
        
        self.condition_encoder = ConditionEncoder(
            condition_mode=self.condition_mode,
            regularization=self.regularization,
            output_dim=self.condition_embedding_dim,
            pooling=self.pooling,
            pooling_kwargs=self.pooling_kwargs,
            layers_before_pool=self.layers_before_pool,
            layers_after_pool=self.layers_after_pool,
            covariates_not_pooled=self.covariates_not_pooled,
            mask_value=self.mask_value,
            **self.condition_encoder_kwargs,
        )

        self.layer_cond_output_dropout = nn.Dropout(rate=self.cond_output_dropout)
        self.layer_norm_condition = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self.layer_norm_time = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_encoder = MLPBlock(
            dims=self.hidden_dims,
            act_fn=self.act_fn,
            dropout_rate=self.hidden_dropout,
            act_last_layer=(not self.linear_projection_before_concatenation),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(not self.linear_projection_before_concatenation),
        )

        self.output_layer = nn.Dense(self.output_dim)

        if self.conditioning == "film":
            self.film_block = FilmBlock(
                input_dim=self.hidden_dims[-1],
                cond_dim=self.time_encoder_dims[-1] + self.condition_embedding_dim,
                **conditioning_kwargs,
            )
        elif self.conditioning == "resnet":
            self.resnet_block = ResNetBlock(
                input_dim=self.hidden_dims[-1],
                **conditioning_kwargs,
            )

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ):
        squeeze = x_t.ndim == 1
        cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        
        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)

        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)

        t_encoded = sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        t_encoded = self.time_encoder(t_encoded, training=train)
        x_encoded = self.x_encoder(x_t, training=train)

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)
        elif cond_embedding.shape[0] != x_t.shape[0]:
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        if self.conditioning == "concatenation":
            out = jnp.concatenate((t_encoded, x_encoded, cond_embedding), axis=-1)
        elif self.conditioning == "film":
            out = self.film_block(x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1))
        elif self.conditioning == "resnet":
            out = self.resnet_block(x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1))
        else:
            raise ValueError(f"Unknown conditioning: {self.conditioning}")

        out = self.decoder(out, training=train)
        return self.output_layer(out), cond_mean, cond_logvar

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]):
        """Get condition embedding (for analysis)."""
        return self.condition_encoder(condition, training=False)

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create training state."""
        t, x_t = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            t=t, x_t=x_t, cond=cond, encoder_noise=encoder_noise, train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

    @property
    def output_dims(self):
        return tuple(self.decoder_dims) + (self.output_dim,)


# ============================================================================
# OT Flow Matching Solver
# ============================================================================

def ema_update(ema_params, new_params, decay):
    """Exponential moving average update."""
    return jax.tree.map(lambda e, n: decay * e + (1 - decay) * n, ema_params, new_params)


class OTFlowMatching:
    """
    OT Flow Matching solver.
    
    Learns a velocity field to transport source distribution to target.
    Uses optimal transport matching and entropy-regularized probability paths.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: dynamics.BaseFlow,
        match_fn: Callable = None,
        time_sampler: Callable = solver_utils.uniform_sampler,
        **kwargs,
    ):
        self._is_trained = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn) if match_fn is not None else None
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

    def _get_vf_step_fn(self):
        @jax.jit
        def vf_step_fn(rng, vf_state, time, source, target, conditions, encoder_noise):
            def loss_fn(params, t, source, target, conditions, encoder_noise, rng):
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params}, t, x_t, conditions, encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                flow_loss = jnp.mean((v_t - u_t) ** 2)
                
                cond_mean_reg = 0.5 * jnp.mean(mean_cond**2)
                cond_var_reg = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = cond_mean_reg + cond_var_reg
                elif self.condition_encoder_mode == "deterministic" and self.condition_encoder_regularization > 0:
                    encoder_loss = cond_mean_reg
                else:
                    encoder_loss = 0.0
                
                total_loss = flow_loss + encoder_loss
                return total_loss, {'flow_loss': flow_loss, 'encoder_loss': encoder_loss}

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, loss_dict), grads = grad_fn(vf_state.params, time, source, target, conditions, encoder_noise, rng)
            return vf_state.apply_gradients(grads=grads), loss, loss_dict

        return vf_step_fn

    def step_fn(self, rng, batch, return_loss_dict=False):
        """Single training step."""
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss, loss_dict = self.vf_step_fn(
            rng_step_fn, self.vf_state, time, src, tgt, condition, encoder_noise
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        
        if return_loss_dict:
            return float(loss), {k: float(v) for k, v in loss_dict.items()}
        return float(loss)

    def predict(self, x, condition, rng=None, **kwargs):
        """Predict target from source using ODE solver."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        if rng is None:
            rng = jax.random.PRNGKey(0)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        def vf(t, x, args):
            params = self.vf_state_inference.params
            cond, enc_noise = args
            return self.vf_state_inference.apply_fn({"params": params}, t, x, cond, enc_noise, train=False)[0]

        def solve_ode(x, condition, encoder_noise):
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term, t0=0.0, t1=1.0, y0=x, args=(condition, encoder_noise), **kwargs
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None, None]))(x, condition, encoder_noise)
        return np.array(x_pred)

    def get_condition_embedding(self, condition, return_as_numpy=True):
        """Get learned condition embeddings."""
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition, method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    @property
    def is_trained(self):
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value):
        self._is_trained = value
