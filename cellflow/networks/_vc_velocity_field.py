import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.image
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from cellflow._types import Layers_separate_input_t, Layers_t
from cellflow.networks._set_encoders import ConditionEncoder
from cellflow.networks._utils import MLPBlock, sinusoidal_time_encoder

__all__ = ["UNet1DVelocityField", "SourceConditionalUNet1DVelocityField"]


def _valid_num_groups(num_channels: int, max_groups: int = 8) -> int:
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class FiLMResBlock1D(nn.Module):
    out_channels: int
    cond_dim: int
    kernel_size: int = 5
    dropout_rate: float = 0.0
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond_vec: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        if residual.shape[-1] != self.out_channels:
            residual = nn.Conv(self.out_channels, kernel_size=(1,), padding="SAME")(residual)

        film = nn.Dense(self.out_channels * 4)(cond_vec)
        gamma1, beta1, gamma2, beta2 = jnp.split(film, 4, axis=-1)

        y = nn.Conv(self.out_channels, kernel_size=(self.kernel_size,), padding="SAME")(x)
        y = nn.GroupNorm(num_groups=_valid_num_groups(self.out_channels))(y)
        y = y * (1.0 + gamma1[:, None, :]) + beta1[:, None, :]
        y = self.act_fn(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)

        y = nn.Conv(self.out_channels, kernel_size=(self.kernel_size,), padding="SAME")(y)
        y = nn.GroupNorm(num_groups=_valid_num_groups(self.out_channels))(y)
        y = y * (1.0 + gamma2[:, None, :]) + beta2[:, None, :]
        y = self.act_fn(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        return residual + y


class Downsample1D(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(self.out_channels, kernel_size=(4,), strides=(2,), padding="SAME")(x)


class Upsample1D(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, output_length: int) -> jnp.ndarray:
        x = jax.image.resize(x, shape=(x.shape[0], output_length, x.shape[-1]), method="nearest")
        return nn.Conv(self.out_channels, kernel_size=(3,), padding="SAME")(x)


class BaseUNet1DVelocityField(nn.Module):
    output_dim: int
    max_combination_length: int
    condition_mode: Literal["deterministic", "stochastic"] = "deterministic"
    regularization: float = 1.0
    condition_embedding_dim: int = 32
    covariates_not_pooled: Sequence[str] = dc_field(default_factory=lambda: [])
    pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token"
    pooling_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: [])
    layers_after_pool: Layers_t = dc_field(default_factory=lambda: [])
    cond_output_dropout: float = 0.0
    mask_value: float = 0.0
    condition_encoder_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    time_freqs: int = 1024
    time_max_period: int = 10000
    time_encoder_dims: Sequence[int] = (512, 512)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (512, 512, 512)
    hidden_dropout: float = 0.0
    conditioning: Literal["concatenation", "film", "resnet"] = "film"
    conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    decoder_dims: Sequence[int] = (512, 512)
    decoder_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False
    unet_channels: Sequence[int] = dc_field(default_factory=lambda: (64, 128, 256))
    unet_kernel_size: int = 5
    enable_mechanism_branch: bool = False
    mechanism_condition_key: str = "mech"
    mechanism_hidden_dim: int = 128
    mechanism_strength: float = 1.0

    def setup(self):
        if isinstance(self.conditioning_kwargs, dataclasses.Field):
            conditioning_kwargs = dict(self.conditioning_kwargs.default_factory())
        else:
            conditioning_kwargs = dict(self.conditioning_kwargs)
        if conditioning_kwargs:
            raise ValueError("UNet1DVelocityField does not support custom `conditioning_kwargs` yet.")

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
        self.time_encoder = MLPBlock(
            dims=self.time_encoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.time_encoder_dropout,
            act_last_layer=False,
        )
        self._resolved_channels = tuple(self._normalize_channels())
        if self.enable_mechanism_branch:
            self.mechanism_source_encoder = MLPBlock(
                dims=(self.mechanism_hidden_dim, self.mechanism_hidden_dim),
                act_fn=self.act_fn,
                dropout_rate=self.hidden_dropout,
                act_last_layer=True,
            )
            self.mechanism_cond_proj = nn.Dense(self.mechanism_hidden_dim)
            self.mechanism_gate = nn.Dense(self.mechanism_hidden_dim)
            self.mechanism_output = nn.Dense(self.output_dim)

    def _normalize_channels(self) -> list[int]:
        if len(self.unet_channels) > 0:
            return [int(ch) for ch in self.unet_channels]

        channels = [max(32, min(256, int(dim // 8))) for dim in self.hidden_dims[:3]]
        while len(channels) < 3:
            next_ch = 64 if len(channels) == 0 else min(256, channels[-1] * 2)
            channels.append(next_ch)
        return channels

    def _encode_condition(
        self,
        t: jnp.ndarray,
        x_like: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)
        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)

        t_encoded = sinusoidal_time_encoder(t, time_freqs=self.time_freqs, time_max_period=self.time_max_period)
        t_encoded = self.time_encoder(t_encoded, training=train)

        if cond_embedding.ndim == 1:
            cond_embedding = cond_embedding[None, :]
        if t_encoded.ndim == 1:
            t_encoded = t_encoded[None, :]
        if cond_embedding.shape[0] != x_like.shape[0]:
            cond_embedding = jnp.tile(cond_embedding, (x_like.shape[0], 1))
        if t_encoded.shape[0] != x_like.shape[0]:
            t_encoded = jnp.tile(t_encoded, (x_like.shape[0], 1))

        global_condition = jnp.concatenate([t_encoded, cond_embedding], axis=-1)
        return cond_mean, cond_logvar, global_condition

    def _run_unet(self, x: jnp.ndarray, global_condition: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        ch1, ch2, ch3 = self._resolved_channels[:3]

        enc1 = FiLMResBlock1D(
            ch1,
            cond_dim=global_condition.shape[-1],
            kernel_size=self.unet_kernel_size,
            dropout_rate=self.hidden_dropout,
            act_fn=self.act_fn,
        )(x, global_condition, train=train)
        down1 = Downsample1D(ch2)(enc1)

        enc2 = FiLMResBlock1D(
            ch2,
            cond_dim=global_condition.shape[-1],
            kernel_size=self.unet_kernel_size,
            dropout_rate=self.hidden_dropout,
            act_fn=self.act_fn,
        )(down1, global_condition, train=train)
        down2 = Downsample1D(ch3)(enc2)

        bottleneck = FiLMResBlock1D(
            ch3,
            cond_dim=global_condition.shape[-1],
            kernel_size=self.unet_kernel_size,
            dropout_rate=self.hidden_dropout,
            act_fn=self.act_fn,
        )(down2, global_condition, train=train)

        up1 = Upsample1D(ch2)(bottleneck, enc2.shape[1])
        up1 = jnp.concatenate([up1, enc2], axis=-1)
        up1 = FiLMResBlock1D(
            ch2,
            cond_dim=global_condition.shape[-1],
            kernel_size=self.unet_kernel_size,
            dropout_rate=self.decoder_dropout,
            act_fn=self.act_fn,
        )(up1, global_condition, train=train)

        up2 = Upsample1D(ch1)(up1, enc1.shape[1])
        up2 = jnp.concatenate([up2, enc1], axis=-1)
        up2 = FiLMResBlock1D(
            ch1,
            cond_dim=global_condition.shape[-1],
            kernel_size=self.unet_kernel_size,
            dropout_rate=self.decoder_dropout,
            act_fn=self.act_fn,
        )(up2, global_condition, train=train)

        return nn.Conv(1, kernel_size=(1,), padding="SAME")(up2)[..., 0]

    def _maybe_apply_mechanism_branch(
        self,
        velocity: jnp.ndarray,
        source_input: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        global_condition: jnp.ndarray,
        train: bool,
    ) -> jnp.ndarray:
        if not self.enable_mechanism_branch:
            return velocity
        if self.mechanism_condition_key not in cond:
            return velocity

        mech = cond[self.mechanism_condition_key]
        if mech.ndim == 3:
            mech = mech.mean(axis=1)
        if mech.ndim == 1:
            mech = mech[None, :]
        if mech.shape[0] != source_input.shape[0]:
            mech = jnp.tile(mech, (source_input.shape[0], 1))

        source_state = self.mechanism_source_encoder(source_input, training=train)
        mech_state = self.mechanism_cond_proj(mech)
        gate_input = jnp.concatenate([source_state, mech_state, global_condition], axis=-1)
        gate = nn.sigmoid(self.mechanism_gate(gate_input))
        mech_bias = self.mechanism_output(gate * mech_state)
        return velocity + self.mechanism_strength * mech_bias

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        condition_mean, condition_logvar = self.condition_encoder(condition, training=False)
        return condition_mean, condition_logvar

    @property
    def output_dims(self):
        return tuple(self._normalize_channels()) + (self.output_dim,)


class UNet1DVelocityField(BaseUNet1DVelocityField):
    @nn.compact
    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x_t.ndim == 1
        if squeeze:
            x_t = x_t[None, :]

        cond_mean, cond_logvar, global_condition = self._encode_condition(t, x_t, cond, encoder_noise, train)
        velocity = self._run_unet(x_t[..., None], global_condition, train=train)
        velocity = self._maybe_apply_mechanism_branch(velocity, x_t, cond, global_condition, train=train)
        if squeeze:
            velocity = jnp.squeeze(velocity, axis=0)
        return velocity, cond_mean, cond_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        t, x_t = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            t=t,
            x_t=x_t,
            cond=cond,
            encoder_noise=encoder_noise,
            train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class SourceConditionalUNet1DVelocityField(BaseUNet1DVelocityField):
    @nn.compact
    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        x_0: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x_t.ndim == 1
        if squeeze:
            x_t = x_t[None, :]
            x_0 = x_0[None, :]

        cond_mean, cond_logvar, global_condition = self._encode_condition(t, x_t, cond, encoder_noise, train)
        stacked_inputs = jnp.stack([x_t, x_0], axis=-1)
        velocity = self._run_unet(stacked_inputs, global_condition, train=train)
        velocity = self._maybe_apply_mechanism_branch(velocity, x_0, cond, global_condition, train=train)
        if squeeze:
            velocity = jnp.squeeze(velocity, axis=0)
        return velocity, cond_mean, cond_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        t = jnp.ones((1, 1))
        x_t = jnp.ones((1, input_dim))
        x_0 = jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            t=t,
            x_t=x_t,
            x_0=x_0,
            cond=cond,
            encoder_noise=encoder_noise,
            train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
