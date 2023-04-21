from jax.tree_util import Partial
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax import jacfwd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import List, Optional, Tuple
from functools import partial

from dynamax.utils.utils import psd_solve
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.types import PRNGKey

# Helper functions
# _get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
# TODO: handle list of parameters, one per timestep
def _get_params(x, dim, t):
    if callable(x):
        return x(t)
    elif x.ndim == dim + 1:
        return x[t]
    else:
        return x
# if inputs==None, assume these functions only take in x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
# if inputs==None, create a sequence of zeros as inputs
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x


def _predict(t, m, P, f, F, Q, u):
    r"""Predict next mean and covariance using first-order additive EKF

        p(z_{t+1}) = \int N(z_t | m, P) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) P F(m, u)^T + Q)

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        F (Callable): Jacobian of dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    F_x = F(t, m, u)
    mu_pred = f(t, m, u)
    Sigma_pred = F_x @ P @ F_x.T + Q
    return mu_pred, Sigma_pred


def _condition_on(t, m, P, h, H, R, u, y, num_iter):
    r"""Condition a Gaussian potential on a new observation.

       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, P) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, PP)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         PP = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         h (Callable): emission function.
         H (Callable): Jacobian of emission function.
         R (D_obs,D_obs): emission covariance matrix.
         u (D_in,): inputs.
         y (D_obs,): observation.
         num_iter (int): number of re-linearizations around posterior for update step.
            Should be a static argument if jitting this function, since it is used in a lax.scan

     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    num_iter = num_iter.shape[0]
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(t, prior_mean, u)
        S = R + H_x @ prior_cov @ H_x.T
        K = psd_solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(t, prior_mean, u))
        return (posterior_mean, posterior_cov), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond

def extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
    state_range: Optional[Tuple[Array, Array]] = None,
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
            Should be a static argument if jitting this function, since it is used in a lax.scan when conditioning
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    num_timesteps = len(emissions)

    inputs = _process_input(inputs, num_timesteps)

    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function  # Assume f(timestep, x) or f(timestep, x, u)
    h = params.emission_function # Assume h(timestep, x) or h(timestep, x, u)
    F = params.dynamics_jacobian if params.dynamics_jacobian is not None else jacfwd(f, 1)
    H = params.emission_jacobian if params.emission_jacobian is not None else jacfwd(h, 1)
    f, h, F, H = (Partial(_process_fn(fn, inputs)) for fn in (f, h, F, H))  # Wrap functions with Partial since it'll be passed through jitted functions 
    # TODO: handle the case where these functions are callable PyTrees

    empty = jnp.empty((num_iter, 0))  # need this to pass num_iter as a static arg to lax.cond

    if state_range is None:
        state_range = (jnp.array([-jnp.inf], dtype=float), jnp.array([jnp.inf], dtype=float))

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry
        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        y = jnp.atleast_1d(y)
        missing = jnp.isnan(y)
        not_missing = ~missing
        R_ = jnp.where(jnp.diag(missing), 1.0, R * (not_missing)[:, None] * not_missing)  # dummy unit variance for missing dimensions, and mask covariances wrt missing dims
        y = jnp.where(missing, jnp.zeros_like(y), y)  # replace with dummy value if missing, to prevent nan gradients
        H_ = lambda *args: H(*args) * (not_missing)[:, None] 
        h_ = lambda *args: jnp.where(missing, 0., h(*args))
        # Update the log likelihood
        H_x = H_(t, pred_mean, u)
        ll += MVN(h_(t, pred_mean, u), (H_x @ pred_cov @ H_x.T + R_)).log_prob(y)
        ll -= -0.5 * jnp.log(2 * jnp.pi) * jnp.sum(missing)  # subtract constant term for missing dimensions


        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(
            t, pred_mean, pred_cov, h_, H_, R_, u, y, empty,
        )
        filtered_mean = jnp.clip(filtered_mean, a_min=state_range[0], a_max=state_range[1])

        # Predict the next state
        pred_mean, pred_cov = _predict(t, filtered_mean, filtered_cov, f, F, Q, u)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            "predicted_means": pred_mean,
            "predicted_covariances": pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def iterated_extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    state_range: Optional[Tuple[Array, Array]] = None,
) -> PosteriorGSSMFiltered:
    r"""Run an iterated extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 2).
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    filtered_posterior = extended_kalman_filter(params, emissions, num_iter, inputs, state_range=state_range)
    return filtered_posterior


def extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    state_range: Optional[Tuple[Array, Array]] = None,
) -> PosteriorGSSMSmoothed:
    r"""Run an extended Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: observation sequence.
        filtered_posterior: optional output from filtering step.
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = extended_kalman_filter(params, emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    f = params.dynamics_function  # Assume f(timestep, x) or f(timestep, x, u)
    F = params.dynamics_jacobian if params.dynamics_jacobian is not None else jacfwd(f, 1)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
    if state_range is None:
        state_range = (-jnp.inf, jnp.inf)

    # Dynamics and emission functions and their Jacobians
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]
        F_x = F(t, filtered_mean, u)

        # Prediction step
        m_pred = f(t, filtered_mean, u)
        S_pred = Q + F_x @ filtered_cov @ F_x.T
        G = psd_solve(S_pred, F_x @ filtered_cov).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T
        smoothed_mean = jnp.clip(smoothed_mean, a_min=state_range[0], a_max=state_range[1])

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the extended Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )


def iterated_extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    num_iter: int = 2,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGSSMSmoothed:
    r"""Run an iterated extended Kalman smoother (IEKS).

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 2).
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """

    def _step(carry, _):
        # Relinearize around smoothed posterior from previous iteration
        smoothed_prior = carry
        smoothed_posterior = extended_kalman_smoother(params, emissions, smoothed_prior, inputs)
        return smoothed_posterior, None

    smoothed_posterior, _ = lax.scan(_step, None, jnp.arange(num_iter))
    return smoothed_posterior

def extended_kalman_posterior_sample(
    key: PRNGKey,
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> Float[Array, "ntime state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples from $p(z_{1:T} \mid y_{1:T}, u_{1:T})$.

    Args:
        key: random number key.
        params: parameters.
        emissions: sequence of observations.
        num_iter (int): number of re-linearizations around posterior for update step.
            Should be static if jitting this function, since it is used in a lax.scan when conditioning
        inputs: optional sequence of inptus.

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $z_{1:T}$ from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Extended Kalman filter
    filtered_posterior = extended_kalman_filter(params, emissions, num_iter, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    f = params.dynamics_function  # Assume f(timestep, x) or f(timestep, x, u)
    F = params.dynamics_jacobian if params.dynamics_jacobian is not None else jacfwd(f, 1)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))

    empty = jnp.empty((num_iter, 0))  # need this to pass num_iter as a static arg to lax.cond
    # Sample backward in time
    def _step(carry, args):
        next_state = carry
        key, filtered_mean, filtered_cov, t = args

        # Shorthand: get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(t, filtered_mean, filtered_cov, f, F, Q, u, next_state, empty)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return state, state

    # Initialize the last state
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    args = (
        jr.split(key, num_timesteps - 1),
        filtered_means[:-1][::-1],
        filtered_covs[:-1][::-1],
        jnp.arange(num_timesteps - 2, -1, -1),
    )
    _, reversed_states = lax.scan(_step, last_state, args)
    states = jnp.row_stack([reversed_states[::-1], last_state])
    return states
