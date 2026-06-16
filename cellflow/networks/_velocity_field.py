import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from cellflow._types import Layers_separate_input_t, Layers_t
from cellflow.networks._set_encoders import ConditionEncoder
from cellflow.networks._utils import FilmBlock, MLPBlock, ResNetBlock, sinusoidal_time_encoder

__all__ = ["ConditionalVelocityField", "GENOTConditionalVelocityField", "PathwayTokenCrossAttn"]


def _condition_encoder_input(cond: dict[str, jnp.ndarray] | None) -> dict[str, jnp.ndarray] | None:
    """Remove CellFlow_VC batch aux keys (``vc_*``) before the set encoder; keep in full ``cond`` for losses."""
    if cond is None:
        return cond
    return {k: v for k, v in cond.items() if not (isinstance(k, str) and k.startswith("vc_"))}


class PathwayTokenCrossAttn(nn.Module):
    """Cross-attention from pooled cell-condition embedding to 24 scalar pathway features."""

    cond_dim: int
    pathway_dim: int = 24
    token_dim: int = 64
    num_heads: int = 1

    @nn.compact
    def __call__(self, pooled_cond: jnp.ndarray, pathway_prior: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        p = jnp.mean(pathway_prior, axis=1)  # (B, 24)
        p_exp = p[..., None]  # (B, 24, 1)
        token_dim = int(self.token_dim)
        d_head = int(token_dim)
        tok = nn.Dense(d_head, name="tok_in")(p_exp)  # (B, 24, d)
        q = nn.Dense(d_head, name="q_from_cond")(pooled_cond)[:, None, :]  # (B, 1, d)
        k = nn.Dense(d_head, name="k_proj")(tok)
        v = nn.Dense(d_head, name="v_proj")(tok)
        # scaled dot-product attention, single "query" per batch item: q (B,1,d) x k (B,24,d) -> (B,1,24)
        scale = 1.0 / jnp.sqrt(jnp.array(float(d_head), dtype=pooled_cond.dtype))
        logits = jnp.matmul(q, jnp.swapaxes(k, 1, 2)) * scale
        w = nn.softmax(logits, axis=-1)  # (B,1,24)
        ctx = jnp.matmul(w, v)[:, 0, :]
        ctx = nn.Dense(self.cond_dim, name="ctx_to_cond")(ctx)
        return pooled_cond + ctx


class ConditionalVelocityField(nn.Module):
    """Parameterized neural vector field with conditions.

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        condition_mode
            Mode of the encoder, should be one of:

            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.

        regularization
            Regularization strength in the latent space:

            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the KL divergence regularization.

        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_max_period
            Controls the minimum frequency of the time embeddings.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        conditioning
            Conditioning method, should be one of:

            - ``'concatenation'``: Concatenate the time, data, and condition embeddings.
            - ``'film'``: Use FiLM conditioning, i.e. learn FiLM weights from time and condition embedding
              to scale the data embeddings.
            - ``'resnet'``: Use residual conditioning.

        conditioning_kwargs
            Keyword arguments for the conditioning method.
        decoder_dims
            Dimensions of the output layers.
        decoder_dropout
            Dropout rate for the output layers.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

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
    # CellFlow_VC (OTFM): optional pathway gating, token cross-attn, aux losses
    use_pathway_source_gate: bool = False
    pathway_source_gate_hidden: int = 256
    use_pathway_token_xattn: bool = False
    pathway_xattn_dim: int = 64
    pathway_xattn_heads: int = 1
    vc_pathway_learned: bool = False
    use_vc_de_head: bool = False
    vc_de_topk: int = 32

    def setup(self):
        """Initialize the network."""
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
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )

        self.output_layer = nn.Dense(self.output_dim)

        if self.use_pathway_source_gate:
            # Gate MLP with hidden layer then a zero-init output so the initial gate is sigmoid(0)=0.5
            # (near-identity multiplier on pathway tokens; prevents NaN at init).
            self.pathway_source_gate_hidden_layer = MLPBlock(
                dims=(int(self.pathway_source_gate_hidden),),
                act_fn=self.act_fn,
                dropout_rate=0.0,
                act_last_layer=True,
            )
            self.pathway_source_gate_ln = nn.LayerNorm()
            self.pathway_source_gate_out = nn.Dense(
                24,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(4.0),  # sigmoid(4) ~= 0.98 => near-identity gate at init
            )

        if self.use_pathway_token_xattn:
            self.pathway_xattn_block = PathwayTokenCrossAttn(
                cond_dim=int(self.condition_embedding_dim),
                pathway_dim=24,
                token_dim=int(self.pathway_xattn_dim),
                num_heads=int(self.pathway_xattn_heads),
            )

        if self.vc_pathway_learned:
            self.pathway_w = self.param(
                "pathway_w", nn.initializers.orthogonal(1.0), (int(self.output_dim), 24)
            )

        if self.use_vc_de_head:
            k2 = 2 * int(self.vc_de_topk)
            self.vc_de_head = MLPBlock(
                dims=(256, k2),
                act_fn=self.act_fn,
                dropout_rate=0.0,
                act_last_layer=True,
            )

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
        elif self.conditioning == "concatenation":
            if len(conditioning_kwargs) > 0:
                raise ValueError("If `conditioning=='concatenation' mode, no conditioning kwargs can be passed.")
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}")

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> (
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ):
        squeeze = x_t.ndim == 1
        cond_in = cond
        if self.use_pathway_source_gate and cond is not None and cond.get("pathway") is not None:
            # Batch-averaged source gate: avoids encoder mask-concat shape mismatch and
            # keeps cond["pathway"]'s batch dim equal to the original (typically 1).
            p = cond["pathway"]  # (1 or B, L, 24)
            x_t_gate = x_t[None, :] if x_t.ndim == 1 else x_t  # (B_or_1, D_lat)
            x_ctx = jnp.mean(x_t_gate, axis=0, keepdims=True)  # (1, D_lat)
            pctx = jnp.mean(p, axis=(0, 1), keepdims=False)[None, :]  # (1, 24)
            g_in = jnp.concatenate([x_ctx, pctx], axis=-1)  # (1, D_lat + 24)
            h = self.pathway_source_gate_hidden_layer(g_in, training=train)
            h = self.pathway_source_gate_ln(h)
            g = nn.sigmoid(self.pathway_source_gate_out(h))  # (1, 24); ~0.98 at init
            g = g[:, None, :]  # (1, 1, 24) broadcasts to (1 or B, L, 24)
            cond_in = {**cond, "pathway": p * g}

        cond_mean, cond_logvar = self.condition_encoder(
            _condition_encoder_input(cond_in), training=train
        )
        if self.use_pathway_token_xattn and cond is not None and cond.get("pathway") is not None:
            cond_mean = self.pathway_xattn_block(cond_mean, cond["pathway"], training=train)
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
            cond_embedding = jnp.squeeze(cond_embedding)  # , 0)
        elif cond_embedding.shape[0] != x_t.shape[0]:  # type: ignore[attr-defined]
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        if self.conditioning == "concatenation":
            out = jnp.concatenate((t_encoded, x_encoded, cond_embedding), axis=-1)
        elif self.conditioning == "film":
            out = self.film_block(x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1))
        elif self.conditioning == "resnet":
            out = self.resnet_block(x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1))
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")

        out = self.decoder(out, training=train)
        v_t = self.output_layer(out)
        if self.vc_pathway_learned:
            v_t = v_t + 0.0 * jnp.sum(self.pathway_w)
        if self.use_vc_de_head:
            de_logits = self.vc_de_head(v_t, training=train)
            return (v_t, cond_mean, cond_logvar, de_logits)
        return (v_t, cond_mean, cond_logvar)

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the embedding of the condition.

        Parameters
        ----------
            condition
                Conditioning vector of shape ``[batch, ...]``.

        Returns
        -------
            Learnt mean and log-variance of the condition embedding.
            If :attr:`cellflow.model.CellFlow.condition_mode` is ``'deterministic'``, the log-variance
            is set to zero.
        """
        cond_in = condition
        if self.use_pathway_source_gate and condition is not None and condition.get("pathway") is not None:
            p = condition["pathway"]
            # No source state available here — use zeros for x context; gate stays near identity at init.
            x_ctx = jnp.zeros((1, int(self.output_dim)), dtype=p.dtype)
            pctx = jnp.mean(p, axis=(0, 1), keepdims=False)[None, :]  # (1, 24)
            g_in = jnp.concatenate([x_ctx, pctx], axis=-1)
            h = self.pathway_source_gate_hidden_layer(g_in, training=False)
            h = self.pathway_source_gate_ln(h)
            g = nn.sigmoid(self.pathway_source_gate_out(h))  # (1, 24)
            g = g[:, None, :]
            cond_in = {**condition, "pathway": p * g}

        condition_mean, condition_logvar = self.condition_encoder(
            _condition_encoder_input(cond_in), training=False
        )
        if self.use_pathway_token_xattn and condition is not None and condition.get("pathway") is not None:
            condition_mean = self.pathway_xattn_block(condition_mean, condition["pathway"], training=False)
        return condition_mean, condition_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create the training state.

        Parameters
        ----------
            rng
                Random number generator.
            optimizer
                Optimizer.
            input_dim
                Dimensionality of the velocity field.
            conditions
                Conditions describing the perturbation.

        Returns
        -------
            The training state.
        """
        t, x_t = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        enc_only = {k: v for k, v in conditions.items() if not (isinstance(k, str) and k.startswith("vc_"))}
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, c.shape[-1])) for pert_cov, c in enc_only.items()
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

    @property
    def output_dims(self):
        """Dimensions of the output layers."""
        return tuple(self.decoder_dims) + (self.output_dim,)

    @property
    def time_encoder(self):
        """The time encoder used."""
        return self._time_encoder

    @time_encoder.setter
    def time_encoder(self, encoder):
        """Set the time encoder."""
        self._time_encoder = encoder

    @property
    def x_encoder(self):
        """The x encoder used."""
        return self._x_encoder

    @x_encoder.setter
    def x_encoder(self, encoder):
        """Set the x encoder."""
        self._x_encoder = encoder

    @property
    def decoder(self):
        """The decoder used."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        """Set the decoder."""
        self._decoder = decoder


class GENOTConditionalVelocityField(ConditionalVelocityField):
    """Parameterized neural vector field with conditions for GENOT.

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        condition_mode
            Mode of the encoder, should be one of:

            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.

        regularization
            Regularization strength in the latent space:

            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the KL divergence regularization.

        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_max_period
            Controls the minimum frequency of the time embeddings.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        conditioning
            Conditioning method, should be one of:

            - ``'concatenation'``: Concatenate the time, data, and condition embeddings.
            - ``'film'``: Use FiLM conditioning, i.e. learn FiLM weights from time, x_0, and condition embedding
              to scale the data embeddings.
            - ``'resnet'``: Use residual conditioning.

        conditioning_kwargs
            Keyword arguments for the conditioning method.
        decoder_dims
            Dimensions of the output layers.
        decoder_dropout
            Dropout rate for the output layers.
        genot_source_dims
            Dimensions of the layers processing the source cells.
        genot_source_dropout
            Dropout rate for the layers processing the source cells.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

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
    time_encoder_dims: Sequence[int] = (1024, 1024, 1024)
    time_encoder_dropout: float = 0.0
    hidden_dims: Sequence[int] = (1024, 1024, 1024)
    hidden_dropout: float = 0.0
    conditioning: Literal["concatenation", "film", "resnet"] = "concatenation"
    conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    decoder_dims: Sequence[int] = (1024, 1024, 1024)
    decoder_dropout: float = 0.0
    genot_source_dims: Sequence[int] = (1024, 1024, 1024)
    genot_source_dropout: float = 0.0
    layer_norm_before_concatenation: bool = False
    linear_projection_before_concatenation: bool = False

    def setup(self):
        """Initialize the network."""
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
            output_dropout=self.cond_output_dropout,
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
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
        )
        self.layer_norm_x = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.x_0_encoder = MLPBlock(
            dims=self.genot_source_dims,
            act_fn=self.act_fn,
            dropout_rate=self.genot_source_dropout,
        )
        self.layer_norm_x_0 = nn.LayerNorm() if self.layer_norm_before_concatenation else lambda x: x

        self.decoder = MLPBlock(
            dims=self.decoder_dims,
            act_fn=self.act_fn,
            dropout_rate=self.decoder_dropout,
            act_last_layer=(False if self.linear_projection_before_concatenation else True),
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
                **self.conditioning_kwargs,
            )
        elif self.conditioning == "concatenation":
            if len(conditioning_kwargs) > 0:
                raise ValueError("If `conditioning=='concatenation' mode, no conditioning kwargs can be passed.")
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}")

    def __call__(
        self,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        x_0: jnp.ndarray,
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
        x_0_encoded = self.x_0_encoder(x_0, training=train)

        t_encoded = self.layer_norm_time(t_encoded)
        x_encoded = self.layer_norm_x(x_encoded)
        x_0_encoded = self.layer_norm_x_0(x_0_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)  # , 0)
        elif cond_embedding.shape[0] != x_t.shape[0]:  # type: ignore[attr-defined]
            cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

        if self.conditioning == "concatenation":
            out = jnp.concatenate((t_encoded, x_encoded, x_0_encoded, cond_embedding), axis=-1)
        elif self.conditioning == "film":
            out = self.film_block(x_encoded, jnp.concatenate((t_encoded, x_0_encoded, cond_embedding), axis=-1))
        elif self.conditioning == "resnet":
            out = self.resnet_block(x_encoded, jnp.concatenate((t_encoded, x_0_encoded, cond_embedding), axis=-1))
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")

        out = self.decoder(out, training=train)
        return self.output_layer(out), cond_mean, cond_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create the training state.

        Parameters
        ----------
            rng
                Random number generator.
            optimizer
                Optimizer.
            input_dim
                Dimensionality of the velocity field.
            conditions
                Conditions describing the perturbation.

        Returns
        -------
            The training state.
        """
        t, x_t, x_0 = jnp.ones((1, 1)), jnp.ones((1, input_dim)), jnp.ones((1, input_dim))
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
