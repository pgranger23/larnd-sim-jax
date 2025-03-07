"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""

import jax.numpy as jnp
from jax.profiler import annotate_function
from jax import jit, lax, random, debug
from jax.lax import stop_gradient
from jax.nn import sigmoid
from jax.scipy.stats import norm
from functools import partial
from larndsim.consts_jax import get_vdrift

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("DETSIM MODULE PARAMETERS")


@partial(jit, static_argnames='signal_length')
def accumulate_signals(wfs, currents_idx, charge, response, pixID, start_ticks, signal_length):
    # Get the number of pixels and ticks
    Npixels, Nticks = wfs.shape

    # Compute indices for updating wfs, taking into account start_ticks
    time_ticks = start_ticks[..., None] + jnp.arange(signal_length)

    time_ticks = jnp.where((time_ticks <= 0 ) | (time_ticks >= wfs.shape[1] - 1), 0, time_ticks+1) # it should be start_ticks +1 in theory but we cheat by putting the cumsum in the garbage too when strarting at 0 to mimic the expected behavior

    start_indices = pixID * Nticks

    end_indices = start_indices[..., None] + time_ticks

    # Flatten the indices
    flat_indices = jnp.ravel(end_indices)

    Nx, Ny, Nt = response.shape

    signal_indices = jnp.ravel((currents_idx[..., 0, None]*Ny + currents_idx[..., 1, None])*Nt + jnp.arange(response.shape[-1] - signal_length, response.shape[-1]))


    # Update wfs with accumulated signals
    wfs = wfs.ravel()
    wfs = wfs.at[(flat_indices,)].add(response.take(signal_indices)*jnp.repeat(charge, signal_length))
    return wfs.reshape((Npixels, Nticks))

@jit
def accumulate_signals_parametrized(wfs, signals, pixID, start_ticks):
    # Get the number of pixels and ticks
    Npixels, Nticks = wfs.shape

    # Compute indices for updating wfs, taking into account start_ticks
    time_ticks = start_ticks[..., None] + jnp.arange(signals.shape[1])

    time_ticks = jnp.where((time_ticks < 0 ) | (time_ticks >= wfs.shape[1] - 1), 0, time_ticks + 1)

    start_indices = pixID * Nticks

    end_indices = start_indices[..., None] + time_ticks

    # Compute indices for updating wfs, taking into account start_ticks
    start_indices = jnp.expand_dims(pixID, axis=1) * Nticks + start_ticks[:, jnp.newaxis]
    end_indices = start_indices + jnp.arange(signals.shape[1])

    # Flatten the indices
    flat_indices = jnp.ravel(end_indices)

    # Update wfs with accumulated signals
    wfs = wfs.ravel()
    wfs = wfs.at[(flat_indices,)].add(signals.ravel())
    return wfs.reshape((Npixels, Nticks))


@annotate_function
@jit
def pixel2id(params, pixel_x, pixel_y, pixel_plane, eventID):
    """
    Convert the x,y,plane tuple to a unique identifier

    Args:
        pixel_x (int): number of pixel pitches in x-dimension
        pixel_y (int): number of pixel pitches in y-dimension
        pixel_plane (int): pixel plane number

    Returns:
        unique integer id
    """
    outside = (pixel_x >= params.n_pixels_x) | (pixel_y >= params.n_pixels_y)
    return jnp.where(outside, -1, pixel_x + params.n_pixels_x * (pixel_y + params.n_pixels_y * (pixel_plane + params.tpc_borders.shape[0]*eventID)))

# @annotate_function
@jit
def id2pixel(params, pid):
    """
    Convert the unique pixel identifer to an x,y,plane tuple

    Args:
        pid (int): unique pixel identifier
    Returns:
        tuple: number of pixel pitches in x-dimension,
            number of pixel pitches in y-dimension,
            pixel plane number
    """
    return (pid % params.n_pixels_x, (pid // params.n_pixels_x) % params.n_pixels_y,
            (pid // (params.n_pixels_x * params.n_pixels_y)) % params.tpc_borders.shape[0],
            pid // (params.n_pixels_x * params.n_pixels_y*params.tpc_borders.shape[0]))

@jit
def get_pixel_coordinates(params, xpitch, ypitch, plane):
    """
    Returns the coordinates of the pixel center given the pixel IDs
    """

    borders = params.tpc_borders[plane.astype(int)]

    pix_x = xpitch  * params.pixel_pitch + borders[..., 0, 0] + params.pixel_pitch/2
    pix_y = ypitch * params.pixel_pitch + borders[..., 1, 0] + params.pixel_pitch/2
    return jnp.stack([pix_x, pix_y], axis=-1)

@jit
def get_hit_z(params, ticks, plane):
    """
    Returns the z position of the hit given the time tick with the right sign for the drift direction
    """
    z_anode = jnp.take(params.tpc_borders, plane.astype(int), axis=0)[..., 2, 0]
    z_high = jnp.take(params.tpc_borders, plane.astype(int), axis=0)[..., 2, 1]
    return z_anode + ticks * params.t_sampling*get_vdrift(params) * jnp.sign(z_high - z_anode)


# @annotate_function
@partial(jit, static_argnames=['fields'])
def generate_electrons(tracks, fields, rngkey):
    sigmas = jnp.stack([tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("long_diff")]], axis=1)
    rnd_pos = random.normal(rngkey, (tracks.shape[0], 3))*sigmas
    electrons = tracks.copy()
    electrons = electrons.at[:, fields.index('x')].set(electrons[:, fields.index('x')] + rnd_pos[:, 0])
    electrons = electrons.at[:, fields.index('y')].set(electrons[:, fields.index('y')] + rnd_pos[:, 1])
    electrons = electrons.at[:, fields.index('z')].set(electrons[:, fields.index('z')] + rnd_pos[:, 2])

    return electrons

# @partial(jit, static_argnames=['num_bins'])
# def adaptive_sampling_gaussian_pdf_3d(sigma, num_bins):
#     """
#     Generate electron packets using a 3D Gaussian PDF with adaptive binning.

#     Args:
#         sigma: Standard deviation of the Gaussian distribution (array of shape (3,)).
#         num_bins: Number of bins for adaptive sampling (same for each dimension).

#     Returns:
#         bin_centers: 3D tuple of bin centers (x, y, z).
#         amplitudes: 3D array of electron amplitudes for each grid point.
#     """
#     # Create adaptive bin edges using the Gaussian CDF for each dimension
#     edges_x = norm.ppf(jnp.linspace(0.001, 0.999, num_bins + 1), loc=0, scale=stop_gradient(sigma[0]))
#     edges_y = norm.ppf(jnp.linspace(0.001, 0.999, num_bins + 1), loc=0, scale=stop_gradient(sigma[1]))
#     edges_z = norm.ppf(jnp.linspace(0.001, 0.999, num_bins + 1), loc=0, scale=stop_gradient(sigma[2]))

#     # Compute bin centers
#     centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
#     centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
#     centers_z = 0.5 * (edges_z[:-1] + edges_z[1:])
#     grid_x, grid_y, grid_z = jnp.meshgrid(centers_x, centers_y, centers_z, indexing='ij')

#     # Evaluate Gaussian PDF at bin centers for each dimension
#     pdf_x = jnp.exp(-0.5 * (grid_x / sigma[0])**2) / (sigma[0] * jnp.sqrt(2 * jnp.pi))
#     pdf_y = jnp.exp(-0.5 * (grid_y / sigma[1])**2) / (sigma[1] * jnp.sqrt(2 * jnp.pi))
#     pdf_z = jnp.exp(-0.5 * (grid_z / sigma[2])**2) / (sigma[2] * jnp.sqrt(2 * jnp.pi))
    
#     # Combine PDFs (assuming independence)
#     pdf = pdf_x * pdf_y * pdf_z

#     # Scale the PDF to match the total number of electrons
#     amplitudes = pdf / jnp.sum(pdf)

#     return (centers_x, centers_y, centers_z), amplitudes

# @partial(jit, static_argnames=['fields'])
# def generate_electrons_distrib(tracks, fields):
    
#     sigmas = jnp.stack([tracks[:, fields.index("tran_diff")],
#                         tracks[:, fields.index("tran_diff")],
#                         tracks[:, fields.index("long_diff")]], axis=1)
#     rnd_pos = random.normal(key, (tracks.shape[0], 3))*sigmas
#     electrons = tracks.copy()
#     electrons = electrons.at[:, fields.index('x')].set(electrons[:, fields.index('x')] + rnd_pos[:, 0])
#     electrons = electrons.at[:, fields.index('y')].set(electrons[:, fields.index('y')] + rnd_pos[:, 1])
#     electrons = electrons.at[:, fields.index('z')].set(electrons[:, fields.index('z')] + rnd_pos[:, 2])

#     return electrons

# @annotate_function
@partial(jit, static_argnames=['fields'])
def get_pixels(params, electrons, fields):
    n_neigh = params.number_pix_neighbors

    borders = params.tpc_borders[electrons[:, fields.index("pixel_plane")].astype(int)]
    pos = jnp.stack([(electrons[:, fields.index("x")] - borders[:, 0, 0]) // params.pixel_pitch,
            (electrons[:, fields.index("y")] - borders[:, 1, 0]) // params.pixel_pitch], axis=1)

    pixels = (pos + 0.5).astype(int)

    X, Y = jnp.mgrid[-n_neigh:n_neigh+1, -n_neigh:n_neigh+1]
    shifts = jnp.vstack([X.ravel(), Y.ravel()]).T
    pixels = pixels[:, jnp.newaxis, :] + shifts[jnp.newaxis, :, :]

    if "eventID" in fields:
        evt_id = "eventID"
    else:
        evt_id = "event_id"

    return pixel2id(params, pixels[:, :, 0], pixels[:, :, 1], electrons[:, fields.index("pixel_plane")].astype(int)[:, jnp.newaxis], electrons[:, fields.index(evt_id)].astype(int)[:, jnp.newaxis])

# @annotate_function
@jit
def truncexpon(x, loc=0, scale=1, y_cutoff=-10., rate=100):
    """
    A truncated exponential distribution.
    To shift and/or scale the distribution use the `loc` and `scale` parameters.
    """
    y = (x - loc) / scale
    # Use smoothed mask to make derivatives nicer
    # y cutoff stops exp from blowing up -- should be far enough away from 0 that sigmoid is small
    y = jnp.maximum(y, y_cutoff)
    return sigmoid(rate*y)*jnp.exp(-y) / scale

# @annotate_function
@jit
def current_model(t, t0, x, y):
    """
    Parametrization of the induced current on the pixel, which depends
    on the of arrival at the anode (:math:`t_0`) and on the position
    on the pixel pad.

    Args:
        t (float): time where we evaluate the current
        t0 (float): time of arrival at the anode
        x (float): distance between the point on the pixel and the pixel center
            on the :math:`x` axis
        y (float): distance between the point on the pixel and the pixel center
            on the :math:`y` axis

    Returns:
        float: the induced current at time :math:`t`
    """
    B_params = (1.060, -0.909, -0.909, 5.856, 0.207, 0.207)
    C_params = (0.679, -1.083, -1.083, 8.772, -5.521, -5.521)
    D_params = (2.644, -9.174, -9.174, 13.483, 45.887, 45.887)
    t0_params = (2.948, -2.705, -2.705, 4.825, 20.814, 20.814)

    a = B_params[0] + B_params[1] * x + B_params[2] * y + B_params[3] * x * y + B_params[4] * x * x + B_params[
        5] * y * y
    b = C_params[0] + C_params[1] * x + C_params[2] * y + C_params[3] * x * y + C_params[4] * x * x + C_params[
        5] * y * y
    c = D_params[0] + D_params[1] * x + D_params[2] * y + D_params[3] * x * y + D_params[4] * x * x + D_params[
        5] * y * y
    shifted_t0 = t0 + t0_params[0] + t0_params[1] * x + t0_params[2] * y + \
                    t0_params[3] * x * y + t0_params[4] * x * x + t0_params[5] * y * y

    a = jnp.minimum(a, 1)

    return a * truncexpon(-t, -shifted_t0, b) + (1 - a) * truncexpon(-t, -shifted_t0, c)

# @annotate_function
@partial(jit, static_argnames=['fields'])
def current_mc(params, electrons, pixels_coord, fields):
    nticks = int(5/params.t_sampling)
    ticks = jnp.linspace(0, 5, nticks).reshape((1, nticks)).repeat(electrons.shape[0], axis=0)#

    x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[..., 0])
    y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[..., 1])
    # signals = jnp.array((electrons.shape[0], ticks.shape[1]))

    z_anode = jnp.take(params.tpc_borders, electrons[:, fields.index("pixel_plane")].astype(int), axis=0)[..., 2, 0]

    t0 = jnp.abs(electrons[:, fields.index('z')] - z_anode) / get_vdrift(params)

    t0_tick = (t0/params.t_sampling + 0.5).astype(int)

    t0 = t0 - t0_tick*params.t_sampling # Only taking the floating part of the ticks

    return t0_tick, current_model(ticks, t0[:, jnp.newaxis], x_dist[:, jnp.newaxis], y_dist[:, jnp.newaxis])*electrons[:, fields.index("n_electrons")].reshape((electrons.shape[0], 1))

@partial(jit, static_argnames=['fields'])
def current_lut(params, response, electrons, pixels_coord, fields):
    x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[..., 0])
    y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[..., 1])
    z_anode = jnp.take(params.tpc_borders, electrons[:, fields.index("pixel_plane")].astype(int), axis=0)[..., 2, 0]
    t0 = jnp.abs(electrons[:, fields.index('z')] - z_anode) / get_vdrift(params)
    
    i = (x_dist/params.response_bin_size).astype(int)
    j = (y_dist/params.response_bin_size).astype(int)


    i = jnp.clip(i, 0, response.shape[0] - 1)
    j = jnp.clip(j, 0, response.shape[1] - 1)

    currents_idx = jnp.stack([i, j], axis=-1)

    return t0, currents_idx
