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
from jax.scipy.special import erfc, erf

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info("DETSIM MODULE PARAMETERS")


@partial(jit, static_argnames='signal_length')
def accumulate_signals(wfs, currents_idx, charge, response, response_cum, pixID, cathode_ticks, signal_length):
    # Get the number of pixels and ticks
    Npixels, Nticks = wfs.shape

    # Compute indices for updating wfs, taking into account start_ticks
    start_ticks = response.shape[-1] - signal_length - cathode_ticks
    time_ticks = start_ticks[..., None] + jnp.arange(signal_length)

    time_ticks = jnp.where((time_ticks <= 0 ) | (time_ticks >= Nticks - 1), 0, time_ticks+1) # it should be start_ticks +1 in theory but we cheat by putting the cumsum in the garbage too when strarting at 0 to mimic the expected behavior

    start_indices = pixID * Nticks

    end_indices = start_indices[..., None] + time_ticks

    # Flatten the indices
    flat_indices = jnp.ravel(end_indices)

    Nx, Ny, Nt = response.shape


    signal_indices = jnp.ravel((currents_idx[..., 0, None]*Ny + currents_idx[..., 1, None])*Nt + jnp.arange(response.shape[-1] - signal_length, response.shape[-1]))
    # baseline_indices = jnp.ravel(jnp.repeat((currents_idx[..., 0]*Ny + currents_idx[..., 1])*Nt + cathode_ticks, signal_length))
    # print(jnp.repeat((currents_idx[..., 0]*Ny + currents_idx[..., 1])*Nt + cathode_ticks, signal_length, axis=0))
    

    # Update wfs with accumulated signals
    wfs = wfs.ravel()
    # wfs = wfs.at[(flat_indices,)].add((response.take(signal_indices) - response.take(baseline_indices))*jnp.repeat(charge, signal_length))
    wfs = wfs.at[(flat_indices,)].add((response.take(signal_indices))*jnp.repeat(charge, signal_length))

    #Now correct for the missed ticks at the beginning
    integrated_start = response_cum.take(jnp.ravel((currents_idx[..., 0]*Ny + currents_idx[..., 1])*Nt + response.shape[-1] - signal_length))
    real_start = response_cum.take(jnp.ravel((currents_idx[..., 0]*Ny + currents_idx[..., 1])*Nt + cathode_ticks))
    difference = (integrated_start - real_start)*charge

    start_ticks = jnp.where((start_ticks <= 0 ) | (start_ticks >= Nticks - 1), 0, start_ticks) + pixID * Nticks
    wfs = wfs.at[start_ticks].add(difference)

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
    outside = (pixel_x >= params.n_pixels_x) | (pixel_y >= params.n_pixels_y) | (pixel_x < 0) | (pixel_y < 0)
    return jnp.where(outside, -1, pixel_x + params.n_pixels_x * (pixel_y + params.n_pixels_y * (pixel_plane + params.tpc_borders.shape[0]*eventID)))

@jit
def bin2id(params, bin_x, bin_y, pixel_plane, eventID):
    """
    Convert the bin coordinates to a unique identifier

    Args:
        bin_x (int): bin coordinate in x-dimension
        bin_y (int): bin coordinate in y-dimension
        pixel_plane (int): pixel plane number
        eventID (int): event identifier

    Returns:
        unique integer id
    """
    outside = (bin_x >= params.n_pixels_x*params.nb_sampling_bins_per_pixel) | (bin_y >= params.n_pixels_y*params.nb_sampling_bins_per_pixel) | (bin_x < 0) | (bin_y < 0)
    return jnp.where(outside, -1, bin_x + params.n_pixels_x*params.nb_sampling_bins_per_pixel * (bin_y + params.n_pixels_y*params.nb_sampling_bins_per_pixel * (pixel_plane + params.tpc_borders.shape[0]*eventID)))

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
def id2bin(params, bin_id):
    """
    Convert the unique bin identifier to an x,y,plane tuple

    Args:
        bin_id (int): unique bin identifier
    Returns:
        tuple: number of pixel pitches in x-dimension,
            number of pixel pitches in y-dimension,
            pixel plane number
    """
    return (bin_id % (params.n_pixels_x*params.nb_sampling_bins_per_pixel), (bin_id // (params.n_pixels_x*params.nb_sampling_bins_per_pixel)) % (params.n_pixels_y*params.nb_sampling_bins_per_pixel),
            (bin_id // ((params.n_pixels_x*params.nb_sampling_bins_per_pixel) * (params.n_pixels_y*params.nb_sampling_bins_per_pixel))) % params.tpc_borders.shape[0],
            bin_id // ((params.n_pixels_x*params.nb_sampling_bins_per_pixel) * (params.n_pixels_y*params.nb_sampling_bins_per_pixel)*params.tpc_borders.shape[0]))

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

# @jit
# def gaussian_1d_integral(bin_edges, std):
#     # Use the error function to compute the integral

#     erf_at_borders = jnp.ones((std.shape[0], bin_edges.shape[0]))
#     erf_at_borders = erf_at_borders.at[:, 0].set(-1)

#     erf_at_borders = erf_at_borders.at[:, 1:-1].set(erf(bin_edges[1:-1] / (jnp.sqrt(2) * std[:, None])))

#     return 0.5*(erf_at_borders[:, 1:] - erf_at_borders[:, :-1])

def gaussian_1d_integral(bin_edges, x0, std):
    # Use the error function to compute the integral

    calculated_edges = bin_edges - x0

    erf_at_borders = jnp.ones_like(calculated_edges)
    erf_at_borders = erf_at_borders.at[:, 0].set(-1)
    erf_at_borders = erf_at_borders.at[:, 1:-1].set(erf(calculated_edges[:, 1:-1] / (jnp.sqrt(2) * std)))

    return 0.5*(erf_at_borders[:, 1:] - erf_at_borders[:, :-1])

@jit
def density_2d(bins, x0, y0, sigma):
    """
    Calculate the 2D density using the Gaussian integral.
    bins: 1D array of bin edges
    x0: 1D array of x0 values
    y0: 1D array of y0 values
    sigma: 1D array of standard deviations
    """
    estimated_x = gaussian_1d_integral(bins[None, :], x0[:, None], sigma[:, None])
    estimated_y = gaussian_1d_integral(bins[None, :], y0[:, None], sigma[:, None])
    return jnp.einsum('ij,ik->ijk', estimated_x, estimated_y)

@partial(jit, static_argnames=['fields'])
def apply_tran_diff(params, electrons, fields):
    # Compute the 1D Gaussian integral for the x dimension
    gaussian_integral_x = gaussian_1d_integral(params.tran_diff_bin_edges, electrons[:, fields.index("tran_diff")])
    gaussian_integral_2d = gaussian_integral_x[:, :, None] * gaussian_integral_x[:, None, :]

    nbins_sym = params.tran_diff_bin_edges.shape[0]/2 - 1

    X, Y = jnp.mgrid[-nbins_sym:nbins_sym+1, -nbins_sym:nbins_sym+1]
    shifts = jnp.vstack([X.ravel(), Y.ravel()]).T
    
   # Foreach electrons entry, apply the possible shifts and give the corresponding weight to the number of electrons using only vectorized operations
    new_electrons = jnp.repeat(electrons[:, None, :], shifts.shape[0], axis=1)
    new_electrons = new_electrons.at[:, :, fields.index("x")].add(shifts[:, 0])
    new_electrons = new_electrons.at[:, :, fields.index("y")].add(shifts[:, 1])
    new_electrons = new_electrons.at[:, :, fields.index("n_electrons")].multiply(gaussian_integral_2d.reshape(-1, shifts.shape[0]))
    return new_electrons.reshape(-1, electrons.shape[1])

# @annotate_function
@partial(jit, static_argnames=['fields', 'apply_long_diffusion'])
def generate_electrons(tracks, fields, rngkey, apply_long_diffusion=True):
    """
    Generate electrons from the tracks.

    Args:
        tracks: Tracks of the particles.
        fields: Fields of the tracks.
        rngkey: Random key for generating noise.
        apply_long_diffusion: Whether to apply longitudinal diffusion.
    Returns:
        electrons: Generated electrons.
    """

    sigmas = jnp.stack([tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("tran_diff")],
                        tracks[:, fields.index("long_diff")]], axis=1)
    rnd_pos = random.normal(rngkey, (tracks.shape[0], 3))*sigmas
    electrons = tracks.copy()
    electrons = electrons.at[:, fields.index('x')].set(electrons[:, fields.index('x')] + rnd_pos[:, 0])
    electrons = electrons.at[:, fields.index('y')].set(electrons[:, fields.index('y')] + rnd_pos[:, 1])
    if apply_long_diffusion:
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


@jit
def emg_pdf(x, mu, sigma, lambd):
    """
    Exponentially Modified Gaussian (EMG) PDF.

    Parameters:
    x (array-like): Input values.
    mu (float): Mean of the Gaussian component.
    sigma (float): Standard deviation of the Gaussian component.
    lambd (float): Rate of the exponential component.

    Returns:
    array-like: EMG PDF values for the input x.
    """
    coeff = lambd / 2
    exponent = coeff * (2 * mu + lambd * sigma**2 - 2 * x)
    erfc_term = erfc((mu + lambd * sigma**2 - x) / (jnp.sqrt(2) * sigma))
    pdf_values = coeff * jnp.exp(exponent) * erfc_term
    return pdf_values


@jit
def integrated_expon_diff(x, loc=0, scale=1, diff=100, dt=1):
    lambd = 1/scale
    a = x - dt/2
    b = x + dt/2

    upper_values = 0.5*erf((b - loc)/(jnp.sqrt(2)*diff)) - emg_pdf(b, loc, diff, lambd)/lambd
    lower_values = 0.5*erf((a - loc)/(jnp.sqrt(2)*diff)) - emg_pdf(a, loc, diff, lambd)/lambd

    tick_values = (upper_values - lower_values)

    tick_values = tick_values/(lower_values[..., 0] - upper_values[..., -1])[..., jnp.newaxis] #Normalize to 1

    return tick_values/dt

# @annotate_function
@partial(jit, static_argnames=['fields'])
def get_pixels(params, electrons, fields):
    n_neigh = params.number_pix_neighbors

    borders = params.tpc_borders[electrons[:, fields.index("pixel_plane")].astype(int)]
    pos = jnp.stack([(electrons[:, fields.index("x")] - borders[:, 0, 0]) // params.pixel_pitch,
            (electrons[:, fields.index("y")] - borders[:, 1, 0]) // params.pixel_pitch], axis=1)

    pixels_int = pos.astype(int)

    X, Y = jnp.mgrid[-n_neigh:n_neigh+1, -n_neigh:n_neigh+1]
    shifts = jnp.vstack([X.ravel(), Y.ravel()]).T
    pixels = pixels_int[:, jnp.newaxis, :] + shifts[jnp.newaxis, :, :]

    if "eventID" in fields:
        evt_id = "eventID"
    else:
        evt_id = "event_id"

    return pixel2id(params, pixels[:, :, 0], pixels[:, :, 1], electrons[:, fields.index("pixel_plane")].astype(int)[:, jnp.newaxis], electrons[:, fields.index(evt_id)].astype(int)[:, jnp.newaxis])

@partial(jit, static_argnames=['fields'])
def get_bin_shifts(params, electrons, fields):
    """
    Compute the bin shifts for the electrons based on their positions.
    Args:
        params: Detector parameters.
        electrons (jnp.ndarray): Array of electrons with fields including 'x', 'y', 'pixel_plane'.
        fields: List of field names in the electrons array.
    Returns:
        bin_shifts (jnp.ndarray): Array of bin shifts for each electron.
    """

    borders = params.tpc_borders[electrons[:, fields.index("pixel_plane")].astype(int)]
    pos = jnp.stack([(electrons[:, fields.index("x")] - borders[:, 0, 0]),
            (electrons[:, fields.index("y")] - borders[:, 1, 0])], axis=1) // (params.pixel_pitch / params.nb_sampling_bins_per_pixel)

    # Compute the bin IDs
    bin_shifts = pos.astype(int)

    return bin_shifts


@partial(jit, static_argnames=['fields'])
def get_bin_id(params, electrons, fields):

    bin_shifts = get_bin_shifts(params, electrons, fields)

    return bin2id(params, bin_shifts[..., 0], bin_shifts[..., 1], electrons[:, fields.index("pixel_plane")].astype(int), electrons[:, fields.index("eventID")].astype(int))


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

@jit
def integrated_expon(x, loc=0, scale=1, rate=100, dt=1):
    return (
        jnp.exp(jnp.minimum(0., (loc - x + dt/2)/scale))
        - jnp.exp(jnp.minimum(0., (loc - x - dt/2)/scale))
        + jnp.exp(jnp.minimum(0., (loc - dt/2)/scale))/x.shape[-1] #Add the last term to make the integral over the whole range 1 shared among all ticks
    )/dt

# @annotate_function
@jit
def current_model(t, t0, x, y, dt):
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

    return a * integrated_expon(-t, -shifted_t0, b, dt=dt) + (1 - a) * integrated_expon(-t, -shifted_t0, c, dt=dt)

@jit
def current_model_diff(t, t0, x, y, dt, sigma):
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

    return a * integrated_expon_diff(-t, -shifted_t0, b, dt=dt, diff=sigma) + (1 - a) * integrated_expon_diff(-t, -shifted_t0, c, dt=dt, diff=sigma)

# @annotate_function
@partial(jit, static_argnames=['fields'])
def current_mc(params, electrons, pixels_coord, fields):
    nticks = int(5/params.t_sampling) + 1
    ticks = jnp.linspace(0, 5, nticks).reshape((1, nticks)).repeat(electrons.shape[0], axis=0)#

    x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[..., 0])
    y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[..., 1])
    # signals = jnp.array((electrons.shape[0], ticks.shape[1]))

    z_anode = jnp.take(params.tpc_borders, electrons[:, fields.index("pixel_plane")].astype(int), axis=0)[..., 2, 0]

    t0 = jnp.abs(electrons[:, fields.index('z')] - z_anode) / get_vdrift(params)

    t0_tick = (t0/params.t_sampling + 0.5).astype(int)

    t0 = t0 - t0_tick*params.t_sampling # Only taking the floating part of the ticks

    dt = 5./(nticks-1)
    if params.diffusion_in_current_sim:
        return t0_tick, current_model_diff(ticks, t0[:, jnp.newaxis], x_dist[:, jnp.newaxis], y_dist[:, jnp.newaxis], dt, electrons[:, fields.index("long_diff")].reshape((electrons.shape[0], 1))/get_vdrift(params))*electrons[:, fields.index("n_electrons")].reshape((electrons.shape[0], 1))
    else:
        return t0_tick, current_model(ticks, t0[:, jnp.newaxis], x_dist[:, jnp.newaxis], y_dist[:, jnp.newaxis], dt)*electrons[:, fields.index("n_electrons")].reshape((electrons.shape[0], 1))

@partial(jit, static_argnames=['fields'])
def current_lut(params, response, electrons, pixels_coord, fields):
    x_dist = abs(electrons[:, fields.index('x')] - pixels_coord[..., 0])
    y_dist = abs(electrons[:, fields.index('y')] - pixels_coord[..., 1])
    z_cathode = jnp.take(params.tpc_borders, electrons[:, fields.index("pixel_plane")].astype(int), axis=0)[..., 2, 1]
    t0 = (jnp.abs(electrons[:, fields.index('z')] - z_cathode)) / get_vdrift(params) #Getting t0 as the equivalent time to cathode
    
    i = (x_dist/params.response_bin_size).astype(int)
    j = (y_dist/params.response_bin_size).astype(int)


    i = jnp.clip(i, 0, response.shape[0] - 1)
    j = jnp.clip(j, 0, response.shape[1] - 1)

    currents_idx = jnp.stack([i, j], axis=-1)

    return t0, currents_idx
