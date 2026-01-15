#!/usr/bin/env python3
"""Quick test script for SIREN surrogate integration."""

import jax
import jax.numpy as jnp
import numpy as np

# Use CPU to avoid GPU memory issues during testing
jax.config.update('jax_platform_name', 'cpu')

from larndsim.consts_jax import build_params_class, load_detector_properties, load_surrogate
from larndsim.sim_jax import simulate_surrogate, simulate_drift_surrogate

print("=" * 60)
print("Testing SIREN Surrogate Integration")
print("=" * 60)

# Build params
print("\n1. Loading detector properties...")
Params = build_params_class([])
params = load_detector_properties(
    Params,
    'src/larndsim/detector_properties/module0.yaml',
    'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml'
)
print(f"   pixel_pitch: {params.pixel_pitch}")
print(f"   response_bin_size: {params.response_bin_size}")

# Load surrogate model from checkpoint
print("\n2. Loading surrogate model...")
checkpoint_path = 'siren_training/seed46/checkpoint_latest.npz'
try:
    surrogate_params, surrogate_apply_fn, params = load_surrogate(checkpoint_path, params)
    print(f"   surrogate_diff_range: {params.surrogate_diff_range}")
    print(f"   surrogate_output_min: {params.surrogate_output_min:.4f}")
    print(f"   surrogate_output_max: {params.surrogate_output_max:.4f}")
except Exception as e:
    print(f"   Error loading model: {e}")
    raise

# Test SIREN evaluation
print("\n3. Testing SIREN evaluation...")
from larndsim.surrogate_utils import normalize_siren_inputs

# Test coordinates: (diff=50, x=22, y=22, t=1000)
test_coords = jnp.array([[50.0, 22.0, 22.0, 1000.0]])
coords_norm = normalize_siren_inputs(
    test_coords[:, 0], test_coords[:, 1], test_coords[:, 2], test_coords[:, 3],
    diff_range=params.surrogate_diff_range,
    x_range=params.surrogate_x_range,
    y_range=params.surrogate_y_range,
    t_range=params.surrogate_t_range,
)
print(f"   Input coords: diff=50, x=22, y=22, t=1000")
print(f"   Normalized: {coords_norm}")

output = surrogate_apply_fn(surrogate_params, coords_norm)
print(f"   SIREN output (raw): {output[0, 0]:.6f}")

# Test coordinate transforms
print("\n4. Testing coordinate transforms...")
from larndsim.surrogate_utils import long_diff_to_siren_diff, position_to_siren_xy

# Test diffusion mapping
long_diff_ticks = jnp.array([0.001, 1.0, 5.0, 10.0])
diff_siren = long_diff_to_siren_diff(long_diff_ticks)
print(f"   long_diff_ticks: {long_diff_ticks}")
print(f"   diff_siren:      {diff_siren}")

# Test position mapping
x_dist = jnp.array([0.0, 0.1, 0.5, 1.0])  # cm
x_frac, y_frac = position_to_siren_xy(x_dist, x_dist, params.response_bin_size)
print(f"   x_dist (cm):     {x_dist}")
print(f"   x_frac (bins):   {x_frac}")

print("\n5. Testing drift surrogate...")
# Create a minimal fake track for testing
fields = ('eventID', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end',
          't_start', 't_end', 'x', 'y', 'z', 't', 'dx', 'dE', 'dEdx', 'n_electrons',
          'long_diff', 'tran_diff', 'pixel_plane', 't0', 'trackID')

# Single test track in TPC 0
tpc0_center = params.tpc_borders[0]
x_pos = float((tpc0_center[0, 0] + tpc0_center[0, 1]) / 2)  # center x
y_pos = float((tpc0_center[1, 0] + tpc0_center[1, 1]) / 2)  # center y
z_pos = float((tpc0_center[2, 0] + tpc0_center[2, 1]) / 2)  # middle of drift

print(f"   Test track position: x={x_pos:.2f}, y={y_pos:.2f}, z={z_pos:.2f}")

# Create track array
track = np.zeros((1, len(fields)), dtype=np.float32)
track[0, fields.index('eventID')] = 0
track[0, fields.index('x_start')] = x_pos
track[0, fields.index('y_start')] = y_pos
track[0, fields.index('z_start')] = z_pos
track[0, fields.index('x_end')] = x_pos + 0.01
track[0, fields.index('y_end')] = y_pos
track[0, fields.index('z_end')] = z_pos
track[0, fields.index('t_start')] = 0.0
track[0, fields.index('t_end')] = 0.0
track[0, fields.index('x')] = x_pos
track[0, fields.index('y')] = y_pos
track[0, fields.index('z')] = z_pos
track[0, fields.index('t')] = 0.0
track[0, fields.index('dx')] = 0.1  # 1mm step
track[0, fields.index('dE')] = 0.2  # MeV (2 MeV/cm * 0.1 cm)
track[0, fields.index('dEdx')] = 2.0  # MeV/cm (typical for MIP)
track[0, fields.index('n_electrons')] = 100000  # More electrons to pass threshold
track[0, fields.index('long_diff')] = 0.01
track[0, fields.index('tran_diff')] = 0.01
track[0, fields.index('pixel_plane')] = 0
track[0, fields.index('t0')] = 0.0
track[0, fields.index('trackID')] = 0

tracks = jnp.array(track)

try:
    result = simulate_drift_surrogate(params, tracks, fields)
    print(f"   simulate_drift_surrogate returned {len(result)} arrays")
    main_pixels, pixels, nelectrons, t0_ticks, long_diff_siren, x_dist_frac, y_dist_frac, *rest = result
    print(f"   main_pixels shape: {main_pixels.shape}")
    print(f"   nelectrons sum: {nelectrons.sum():.1f}")
    print(f"   t0_ticks range: [{t0_ticks.min():.1f}, {t0_ticks.max():.1f}]")
    print(f"   long_diff_siren: {long_diff_siren[:5]}")
    print(f"   x_dist_frac range: [{x_dist_frac.min():.2f}, {x_dist_frac.max():.2f}]")
except Exception as e:
    print(f"   Error in simulate_drift_surrogate: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Testing full surrogate simulation...")
try:
    result = simulate_surrogate(params, surrogate_params, surrogate_apply_fn, tracks, fields, save_wfs=True)
    adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels, q_cumsum = result
    print(f"   Output shapes:")
    print(f"     adcs: {adcs.shape}")
    print(f"     ticks: {ticks.shape}")
    print(f"     q_cumsum: {q_cumsum.shape}")
    print(f"   ADC values: {adcs[:5]}")
    print(f"   Ticks: {ticks[:5]}")
    print(f"   q_cumsum max: {q_cumsum.max():.2f}")
except Exception as e:
    print(f"   Error in simulate_surrogate: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
