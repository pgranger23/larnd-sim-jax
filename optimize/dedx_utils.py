"""
Per-segment dEdx fitting utilities.

Provides the Student-t NLL prior used to regularise per-segment dEdx parameters
during joint fitting, plus default constants for the LAr minimum-ionising-particle
dEdx distribution.
"""

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Default Student-t prior parameters for dEdx in LAr  [MeV/cm]
# These were fitted from the minimum-ionising-particle dEdx distribution
# using scipy.stats.t.fit on simulated tracks.  Override via the
# dedx_student_{nu,loc,scale} arguments to GradientDescentFitter.
# ---------------------------------------------------------------------------
DEDX_STUDENT_NU    = jnp.array(2.715,  dtype=jnp.float32)   # degrees of freedom
DEDX_STUDENT_LOC   = jnp.array(1.864,  dtype=jnp.float32)   # location (MeV/cm)
DEDX_STUDENT_SCALE = jnp.array(0.111,  dtype=jnp.float32)   # scale     (MeV/cm)


def student_t_nll(x, df, loc, scale, weights=None):
    """Negative log-likelihood of the Student-t distribution.

    The constant log-gamma terms are omitted because they do not affect the
    gradient with respect to ``x``.

    Parameters
    ----------
    x       : jax array  — samples (e.g. per-step fitted dEdx values)
    df      : scalar     — degrees of freedom  (ν > 0)
    loc     : scalar     — location parameter
    scale   : scalar     — scale parameter     (σ > 0)
    weights : jax array  — weights for each sample (e.g. step length dx). 
                           If None, performs an unweighted mean.

    Returns
    -------
    Scalar NLL. If weights are provided, returns the weighted average.
    """
    z   = (x - loc) / scale
    nll = jnp.log(scale) + 0.5 * (df + 1.0) * jnp.log1p(z * z / df)
    
    if weights is not None:
        # Mask out invalid segments (weight <= 0)
        valid_mask = (weights > 0)
        return jnp.sum(nll * weights * valid_mask)
    else:
        return jnp.sum(nll)
