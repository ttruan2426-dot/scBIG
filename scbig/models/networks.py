"""
Network building blocks for scBIG.

- MLPBlock, FilmBlock, ResNetBlock
- Attention modules (SelfAttention, pooling)
- Sinusoidal time encoding
"""

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers


def sinusoidal_time_encoder(t: jnp.ndarray, time_freqs: int = 1024, time_max_period: int = 10000) -> jnp.ndarray:
    """Sinusoidal timestep embedding."""
    t = t * time_max_period
    half = time_freqs // 2
    freqs = jnp.exp(-math.log(time_max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
    args = t * freqs
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding


class MLPBlock(nn.Module):
    """MLP block with dropout."""
    dims: Sequence[int] = (1024, 1024, 1024)
    dropout_rate: float = 0.0
    act_last_layer: bool = True
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if len(self.dims) == 0:
            return x
        z = x
        for i in range(len(self.dims) - 1):
            z = self.act_fn(nn.Dense(self.dims[i])(z))
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
        z = nn.Dense(self.dims[-1])(z)
        z = self.act_fn(z) if self.act_last_layer else z
        z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
        return z


class FilmBlock(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer."""
    input_dim: int
    cond_dim: int
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x

    def setup(self) -> None:
        self.film_generator = nn.Dense(self.input_dim * 2)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        gamma_beta = self.film_generator(cond)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
        return self.act_fn(gamma * x + beta)


class ResNetBlock(nn.Module):
    """Residual conditioning block."""
    input_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    projection_dims: Sequence[int] = (256, 256)
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    dropout_rate: float = 0.0

    def setup(self):
        self.mlp_block_1 = MLPBlock(dims=self.hidden_dims, act_fn=self.act_fn, dropout_rate=self.dropout_rate)
        self.mlp_block_2 = MLPBlock(
            dims=list(self.hidden_dims) + [self.input_dim], act_fn=self.act_fn, dropout_rate=self.dropout_rate
        )
        self.cond_proj = MLPBlock(dims=self.projection_dims, act_fn=self.act_fn, dropout_rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, *, training: bool = True) -> jnp.ndarray:
        h = self.mlp_block_1(x)
        h = h + self.cond_proj(cond)
        h = self.mlp_block_2(h)
        return h + x


class SelfAttention(nn.Module):
    """Self-attention layer."""
    num_heads: int = 8
    qkv_dim: int = 64
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = True):
        squeeze = x.ndim == 2
        x = jnp.expand_dims(x, 1) if squeeze else x

        z = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            dropout_rate=self.dropout_rate,
        )(x, mask=mask, deterministic=not training)

        if self.transformer_block:
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
            z = z + x
            if self.layer_norm:
                z = nn.LayerNorm()(z)
            z_ = self.act_fn(nn.Dense(self.qkv_dim)(z))
            z_ = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
            z = z + z_

        return z.squeeze(1) if squeeze else z


class SelfAttentionBlock(nn.Module):
    """Stacked self-attention layers."""
    num_heads: Sequence[int] | int
    qkv_dim: Sequence[int] | int
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.num_heads, Sequence):
            self.num_heads = [self.num_heads]
        if not isinstance(self.qkv_dim, Sequence):
            self.qkv_dim = [self.qkv_dim]

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = True) -> jnp.ndarray:
        z = x
        for num_heads, qkv_dim in zip(self.num_heads, self.qkv_dim, strict=False):
            z = SelfAttention(
                num_heads=num_heads,
                qkv_dim=qkv_dim,
                dropout_rate=self.dropout_rate,
                transformer_block=self.transformer_block,
                layer_norm=self.layer_norm,
                act_fn=self.act_fn,
            )(z, mask, training)
        return z


class SeedAttentionPooling(nn.Module):
    """Pooling by attention with trainable seed."""
    num_heads: int = 8
    v_dim: int = 64
    seed_dim: int = 64
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = True):
        S = self.param("S", initializers.xavier_uniform(), (1, 1, self.seed_dim))
        S = jnp.tile(S, (x.shape[0], 1, 1))

        Q = nn.Dense(self.v_dim)(S)
        K, V = nn.Dense(self.v_dim)(x), nn.Dense(self.v_dim)(x)
        Q_ = jnp.concatenate(jnp.split(Q, self.num_heads, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(K, self.num_heads, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(V, self.num_heads, axis=2), axis=0)
        A = jnp.matmul(Q_, K_.transpose(0, 2, 1)) / jnp.sqrt(self.v_dim)
        
        if mask is not None:
            mask = jnp.repeat(mask[:, 0, [0], :], self.num_heads, axis=0)
            A = jnp.where(mask, A, -1e9)
        A = nn.softmax(A)
        A = jnp.matmul(A, V_)

        if self.transformer_block:
            O = jnp.concatenate(jnp.split(Q_ + A, self.num_heads, axis=0), axis=2)
            O = nn.Dropout(rate=self.dropout_rate)(O, deterministic=not training)
            if self.layer_norm:
                O = nn.LayerNorm()(O)
            O_ = self.act_fn(nn.Dense(self.v_dim)(O))
            O_ = nn.Dropout(rate=self.dropout_rate)(O_, deterministic=not training)
            O = O + O_
            if self.layer_norm:
                O = nn.LayerNorm()(O)
        else:
            O = jnp.concatenate(jnp.split(A, self.num_heads, axis=0), axis=2)

        return O.squeeze(1)


class TokenAttentionPooling(nn.Module):
    """Pooling by learning a token."""
    num_heads: int = 8
    qkv_dim: int = 64
    dropout_rate: float = 0.0
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None, training: bool = True) -> jnp.ndarray:
        token_shape = (len(x), 1)
        class_token = nn.Embed(num_embeddings=1, features=x.shape[-1])(jnp.int32(jnp.zeros(token_shape)))
        z = jnp.concatenate((class_token, x), axis=-2)
        
        token_mask = jnp.ones((x.shape[0], 1, x.shape[1] + 1, x.shape[1] + 1))
        token_mask = token_mask.at[:, :, 1:, 1:].set(mask)
        cls_token_to_data = mask[0, 0, :, :].sum(axis=0) > 0
        token_mask = token_mask.at[:, :, 0, 1:].set(cls_token_to_data)
        token_mask = token_mask.at[:, :, 1:, 0].set(cls_token_to_data)

        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            dropout_rate=self.dropout_rate,
        )
        emb = attention(z, mask=token_mask, deterministic=not training)
        z = emb[:, 0, :]
        return z
