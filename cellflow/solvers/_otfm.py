from collections.abc import Callable
from functools import partial
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import frozen_dict
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers.utils import ema_update

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        probability_path
            Probability path between the source and the target distributions.
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`cellflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        kwargs
            Keyword arguments for :meth:`cellflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn) if match_fn is not None else None
        self.ema = kwargs.pop("ema", 1.0)
        
        # Direction loss hyperparameters (for latent-space direction supervision)
        self.dir_lambda_latent = kwargs.pop("dir_lambda_latent", 0.0)  # Default 0.0 = disabled
        self.dir_topk_latent = kwargs.pop("dir_topk_latent", 20)
        self.dir_tau_latent = kwargs.pop("dir_tau_latent", 3.0)
        # Decoded-space DIR regression (e.g., PCA latent -> gene delta)
        self.dir_lambda_decoded = kwargs.pop("dir_lambda_decoded", 0.0)
        self.dir_topk_genes = kwargs.pop("dir_topk_genes", 200)
        dir_decoder_matrix = kwargs.pop("dir_decoder_matrix", None)
        self.dir_decoder_matrix = None if dir_decoder_matrix is None else jnp.asarray(dir_decoder_matrix)

        # CellFlow_VC: pathway activity consistency and pseudo-DE / DIR auxiliary losses
        self.vc_pathway_mode = str(kwargs.pop("vc_pathway_mode", "off"))  # off | learned | matrix
        self.vc_pathway_loss_w = float(kwargs.pop("vc_pathway_loss_w", 0.0))
        vc_pathway_k_latent = kwargs.pop("vc_pathway_k_latent", None)
        self.vc_pathway_k_latent = None if vc_pathway_k_latent is None else jnp.asarray(vc_pathway_k_latent, dtype=jnp.float32)
        self.vc_de_loss_w = float(kwargs.pop("vc_de_loss_w", 0.0))

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

    def _compute_direction_loss_latent(
        self,
        pred_delta: jnp.ndarray,
        true_delta: jnp.ndarray,
        topk: int,
        temperature: float,
    ) -> jnp.ndarray:
        """
        Compute direction BCE loss on top-K dimensions in latent space.
        
        Args:
            pred_delta: (batch, latent_dim) predicted velocity/direction
            true_delta: (batch, latent_dim) true delta (target - source)
            topk: number of top dimensions to focus on (must be static for JIT)
            temperature: temperature for logits
            
        Returns:
            Scalar direction loss
        """
        batch_size, latent_dim = true_delta.shape
        
        # Compute per-sample, per-dim direction loss
        # For each dimension: if true_delta > 0, we want pred_delta > 0 (and vice versa)
        true_sign = (true_delta > 0).astype(jnp.float32)  # 1 if positive, 0 if negative
        logits = pred_delta / temperature
        
        # BCEWithLogits: loss = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
        # JAX version: optax.sigmoid_binary_cross_entropy
        # Equivalent: log(1 + exp(-logits)) if y=1, log(1 + exp(logits)) if y=0
        bce_per_dim = jnp.where(
            true_sign > 0.5,
            jnp.log1p(jnp.exp(-logits)),  # y=1
            jnp.log1p(jnp.exp(logits))     # y=0
        )  # (batch, latent_dim)
        
        # Weight by |true_delta| and select top-K per sample
        abs_delta = jnp.abs(true_delta)  # (batch, latent_dim)
        
        # For each sample, select top-K dimensions
        # IMPORTANT: k must be a Python int (not JAX array) for JIT compilation
        k = min(int(topk), int(latent_dim))  # Use Python min to get static int
        _, top_indices = jax.lax.top_k(abs_delta, k)  # (batch, k)
        
        # Gather BCE values for top-K dimensions
        # Use vmap to handle batch dimension
        def select_topk_for_sample(bce_row, indices):
            return bce_row[indices]
        
        bce_topk = jax.vmap(select_topk_for_sample)(bce_per_dim, top_indices)  # (batch, k)
        
        # Average over top-K dimensions and batch
        return jnp.mean(bce_topk)

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                out_vf = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                if len(out_vf) == 4:
                    v_t, mean_cond, logvar_cond, de_logits = out_vf
                else:
                    v_t, mean_cond, logvar_cond = out_vf
                    de_logits = None
                # ott-jax 0.4.x expects compute_ut(t, x_t, x0, x1).
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                
                # Direction loss in latent space (if enabled)
                if self.dir_lambda_latent > 0:
                    true_delta = target - source  # (batch, latent_dim)
                    pred_delta = v_t  # Use velocity field as direction prediction
                    # Convert to Python int/float to ensure static values for JIT
                    topk_static = int(self.dir_topk_latent)
                    tau_static = float(self.dir_tau_latent)
                    dir_loss_latent = self._compute_direction_loss_latent(
                        pred_delta, true_delta, topk_static, tau_static
                    )
                else:
                    dir_loss_latent = 0.0

                if self.dir_lambda_decoded > 0 and self.dir_decoder_matrix is not None:
                    pred_delta_gene = v_t @ self.dir_decoder_matrix
                    true_delta_gene = (target - source) @ self.dir_decoder_matrix
                    abs_true_delta_gene = jnp.abs(true_delta_gene)
                    n_genes = abs_true_delta_gene.shape[1]
                    k_genes = max(1, min(int(self.dir_topk_genes), int(n_genes)))
                    _, top_gene_idx = jax.lax.top_k(abs_true_delta_gene, k_genes)

                    def select_topk(row, idx):
                        return row[idx]

                    pred_top = jax.vmap(select_topk)(pred_delta_gene, top_gene_idx)
                    true_top = jax.vmap(select_topk)(true_delta_gene, top_gene_idx)
                    dir_loss_decoded = jnp.mean((pred_top - true_top) ** 2)
                else:
                    dir_loss_decoded = 0.0

                pathway_loss = 0.0
                if (
                    self.vc_pathway_mode in ("learned", "matrix")
                    and self.vc_pathway_loss_w > 0.0
                    and conditions is not None
                    and "pathway" in conditions
                    and conditions.get("pathway") is not None
                ):
                    pw = conditions["pathway"]
                    prior = jnp.mean(pw, axis=1, dtype=source.dtype)  # (B, 24) or (1, 24)
                    bsz = int(source.shape[0])
                    if int(prior.shape[0]) == 1 and bsz > 1:
                        prior = jnp.broadcast_to(prior, (bsz, int(prior.shape[1])))
                    delta_lat = target - source
                    w_learned = params.get("pathway_w")  # type: ignore[union-attr]
                    if self.vc_pathway_mode == "learned" and w_learned is not None:
                        delta_act = delta_lat @ w_learned
                        pathway_loss = jnp.mean((delta_act - prior) ** 2)
                    elif self.vc_pathway_mode == "matrix" and self.vc_pathway_k_latent is not None:
                        delta_act = delta_lat @ self.vc_pathway_k_latent
                        pathway_loss = jnp.mean((delta_act - prior) ** 2)
                    else:
                        pathway_loss = 0.0
                else:
                    pathway_loss = 0.0

                de_aux_loss = 0.0
                if (
                    de_logits is not None
                    and self.vc_de_loss_w > 0.0
                    and conditions is not None
                    and "vc_de_binary" in conditions
                    and "vc_de_dir" in conditions
                ):
                    k = de_logits.shape[1] // 2
                    yb = conditions["vc_de_binary"]  # shape (1 or B, L, topk)
                    yd = conditions["vc_de_dir"]    # shape (1 or B, L, topk)
                    if yb is not None and yd is not None:
                        # collapse the set-dim L by taking slot 0 (labels are identical across L by construction)
                        yb = yb[..., 0, :] if yb.ndim == 3 else yb  # (1 or B, topk)
                        yd = yd[..., 0, :] if yd.ndim == 3 else yd
                        bsz = de_logits.shape[0]
                        if yb.shape[0] == 1 and bsz > 1:
                            yb = jnp.broadcast_to(yb, (bsz, yb.shape[-1]))
                        if yd.shape[0] == 1 and bsz > 1:
                            yd = jnp.broadcast_to(yd, (bsz, yd.shape[-1]))
                        bce1 = optax.sigmoid_binary_cross_entropy(de_logits[:, :k], yb)
                        bce2 = optax.sigmoid_binary_cross_entropy(de_logits[:, k:], yd)
                        de_aux_loss = jnp.mean(bce1) + jnp.mean(bce2)
                
                total_loss = (
                    flow_matching_loss
                    + encoder_loss
                    + self.dir_lambda_latent * dir_loss_latent
                    + self.dir_lambda_decoded * dir_loss_decoded
                    + self.vc_pathway_loss_w * pathway_loss
                    + self.vc_de_loss_w * de_aux_loss
                )
                
                # Return loss components for logging
                loss_dict = {
                    "flow_matching_loss": flow_matching_loss,
                    "encoder_loss": encoder_loss,
                    "dir_loss_latent": dir_loss_latent,
                    "dir_loss_decoded": dir_loss_decoded,
                    "pathway_consistency_loss": pathway_loss,
                    "de_aux_loss": de_aux_loss,
                    "total_loss": total_loss,
                }
                return total_loss, loss_dict

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, loss_dict), grads = grad_fn(vf_state.params, time, source, target, conditions, encoder_noise, rng)
            return vf_state.apply_gradients(grads=grads), loss, loss_dict

        return vf_step_fn

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
        return_loss_dict: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            optionally ``condition``.
        return_loss_dict
            If True, return (loss, loss_dict) with loss components for logging.

        Returns
        -------
        Loss value, or (loss, loss_dict) if return_loss_dict=True.
        """
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))
        # TODO: test whether it's better to sample the same noise for all samples or different ones

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss, loss_dict = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            src,
            tgt,
            condition,
            encoder_noise,
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        
        if return_loss_dict:
            # Convert JAX arrays to Python floats for logging
            loss_dict_float = {k: float(v) for k, v in loss_dict.items()}
            return float(loss), loss_dict_float
        return float(loss)

    def get_condition_embedding(self, condition: dict[str, ArrayLike], return_as_numpy=True) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.

        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def _predict_jit(
        self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None = None, **kwargs: Any
    ) -> ArrayLike:
        """See :meth:`OTFlowMatching.predict`."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params = self.vf_state_inference.params
            condition, encoder_noise = args
            return self.vf_state_inference.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]

        def solve_ode(x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=(condition, encoder_noise),
                **kwargs,
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None, None]))(x, condition, encoder_noise)
        return x_pred

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            A dictionary with keys indicating the name of the condition and values containing
            the input data as arrays. If ``batched=False`` provide an array of shape [batch_size, ...].
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition of input data as arrays. If ``batched=False`` provide an array of shape
            [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        batched
            Whether to use batched prediction. This is only supported if the input has
            the same number of cells for each condition. For example, this works when using
            :class:`~cellflow.data.ValidationSampler` to sample the validation data.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if batched and not x:
            return {}

        if batched:
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            _predict_jit = jax.jit(lambda x, condition: self._predict_jit(x, condition, rng, **kwargs))
            batched_predict = jax.vmap(_predict_jit, in_axes=(0, dict.fromkeys(condition_keys, 0)))
            # assert that the number of cells is the same for each condition
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert x[k].shape[0] == n_cells, "The number of cells must be the same for each condition"
            src_inputs = jnp.stack([x[k] for k in keys], axis=0)
            batched_conditions = {}
            for cond_key in condition_keys:
                batched_conditions[cond_key] = jnp.stack([condition[k][cond_key] for k in keys])

            pred_targets = batched_predict(src_inputs, batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        elif isinstance(x, dict):
            return jax.tree.map(
                partial(self._predict_jit, rng=rng, **kwargs),
                x,
                condition,  # type: ignore[attr-defined]
            )
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
