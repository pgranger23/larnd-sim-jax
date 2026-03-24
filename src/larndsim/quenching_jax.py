"""
Module to implement the quenching of the ionized electrons
through the detector
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from larndsim.consts_jax import RecombinationMode

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("QUENCHING MODULE PARAMETERS")

def box_model(dEdx, eField, lArDensity, alpha, beta):
    # Baller, 2013 JINST 8 P08005
    csi = beta * dEdx / (eField * lArDensity)
    return jnp.maximum(0, jnp.log(alpha + csi) / csi)

def birks_model(dEdx, eField, lArDensity, Ab, kb):
    # Amoruso, et al NIM A 523 (2004) 275
    return Ab / (1 + kb * dEdx / (eField * lArDensity))

def get_nelectrons(dE, recomb, MeVToElectrons):
    return recomb * dE * MeVToElectrons

def ellipsoid_box_model(dEdx, cosphi, eField, lArDensity, alpha, beta, R_param):
    # ICARUS EMB Model incorporating track angle dependence
    # b_phi represents the angular-dependent B(phi) parameter
    b_phi = beta / jnp.sqrt(1 - cosphi**2 + (1.0 / R_param**2) * cosphi**2)
    csi = b_phi * dEdx / (eField * lArDensity)
    return jnp.maximum(0, jnp.log(alpha + csi) / (csi + 1e-10))

@partial(jit, static_argnames=['fields'])
def quench(params, tracks, fields):
    """
    This function takes as input an (unstructured) array of track segments and calculates
    the number of electrons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).

    Args:
        tracks (:obj:`numpy.ndarray`, `JAX Tensor`): array containing the tracks segment information
        mode (RecombinationMode): recombination model.
        fields (list): an ordered string list of field/column name of the tracks structured array
    """

    if params.recombination_mode == RecombinationMode.BOX:
        recomb = box_model(tracks[:, fields.index("dEdx")], params.eField, params.lArDensity, params.alpha, params.beta)
    elif params.recombination_mode == RecombinationMode.BIRKS:
        recomb = birks_model(tracks[:, fields.index("dEdx")], params.eField, params.lArDensity, params.Ab, params.kb)
    elif params.recombination_mode == RecombinationMode.ELLIPSOID:
        # Requires the track angle 'phi' (in radians) to be present in the tracks array
        cosphi = jnp.abs(tracks[:, fields.index("z_end")] - tracks[:, fields.index("z_start")]) / (tracks[:, fields.index("dx")] + 1e-10)
        recomb = ellipsoid_box_model(
            tracks[:, fields.index("dEdx")], 
            cosphi, 
            params.eField, 
            params.lArDensity, 
            params.alpha, 
            params.beta, 
            params.R_param
        )
    else:
        raise ValueError(f"Invalid recombination mode {params.recombination_mode}: must be 'BOX', 'BIRKS', or 'ELLIPSOID'")

    #TODO: n_electrons should be int, but truncation makes gradients vanish
    updated_tracks = tracks.at[:, fields.index("n_electrons")].set(get_nelectrons(tracks[:, fields.index("dE")], recomb, params.MeVToElectrons))
    return updated_tracks