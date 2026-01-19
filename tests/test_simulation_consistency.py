import h5py
import numpy as np
import argparse
import sys
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from larndsim.consts_jax import build_params_class, load_detector_properties
from larndsim.sim_jax import id2pixel, get_pixel_coordinates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_results(fname):
    with h5py.File(fname, 'r') as f:
        if 'pixels' in f.keys():
            pixels = np.array(f['pixels'])
            if 'adc_clean' in f.keys():
                adc = np.array(f['adc_clean'])
            else:
                adc = np.array(f['adc'])
        else:
            # Handle structured H5 if necessary, taking first key
            key = list(f.keys())[0]
            pixels = np.array(f[key]['pixels'])
            if 'adc_clean' in f[key].keys():
                adc = np.array(f[key]['adc_clean'])
            else:
                adc = np.array(f[key]['adc'])
    return pixels, adc

def make_grids(pixels, adcs, ref_params):
    xpitch, ypitch, plane, eid = id2pixel(ref_params, pixels)
    coords = get_pixel_coordinates(ref_params, xpitch, ypitch, np.zeros(pixels.shape[0]))
    grids = []
    for p in [0, 1]:
        xbins = np.linspace(ref_params.tpc_borders[p][0][0], ref_params.tpc_borders[p][0][1], ref_params.n_pixels_x + 1)
        ybins = np.linspace(ref_params.tpc_borders[p][1][0], ref_params.tpc_borders[p][1][1], ref_params.n_pixels_y + 1)
        
        if adcs.ndim == 2:
            w = np.sum(adcs - adcs[0, -1], axis=1)[p == plane]
        else:
            w = adcs[p == plane]

        grid, xbins, ybins = np.histogram2d(coords[p == plane][:, 0], coords[p == plane][:, 1], bins=(xbins, ybins), weights=w)
        grids.append(grid)
    return grids

def check_simulation_consistency(ref_file, test_file, config):
    logger.info(f"Comparing {test_file} against {ref_file}")

    if not Path(ref_file).exists():
        logger.error(f"Reference file {ref_file} not found.")
        return False
    if not Path(test_file).exists():
        logger.error(f"Test file {test_file} not found.")
        return False

    Params = build_params_class([])
    ref_params = load_detector_properties(Params, config.detector_props, config.pixel_layouts)

    pixels_ref, adcs_ref = load_results(ref_file)
    grids_ref = make_grids(pixels_ref, adcs_ref, ref_params)

    pixels_new, adcs_new = load_results(test_file)
    grids_new = make_grids(pixels_new, adcs_new, ref_params)

    all_diffs = []
    for grid_ref, grid_new in zip(grids_ref, grids_new):
        mask = (grid_ref > 0) | (grid_new > 0)
        diff = grid_new - grid_ref
        all_diffs.append(diff[mask])
    
    all_diffs = np.concatenate(all_diffs)
    
    mean_diff = np.mean(all_diffs)
    rms_diff = np.std(all_diffs)
    max_diff = np.max(np.abs(all_diffs))
    
    logger.info(f"Mean Diff: {mean_diff:.4e}")
    logger.info(f"RMS Diff: {rms_diff:.4e}")
    logger.info(f"Max Diff: {max_diff:.4e}")

    # Plotting
    test_fname = Path(test_file).stem
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(all_diffs, bins=50, histtype='step', log=True, label="Diff (New - Ref)")
    plt.xlabel("ADC Difference")
    plt.ylabel("Count")
    plt.title(f"ADC Difference Distribution: {test_fname}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"diff_hist_{test_fname}.png")
    plt.close()

    # 2D Grid Plots (using first plane for brevity)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.pcolormesh(grids_ref[0].T) # Transpose to match x-y orientation usually
    plt.title(f"Reference (Plane 0)")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.pcolormesh(grids_new[0].T)
    plt.title(f"New (Plane 0)")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    diff_grid = grids_new[0] - grids_ref[0]
    plt.pcolormesh(diff_grid.T, cmap='RdBu', vmin=-np.max(np.abs(diff_grid)), vmax=np.max(np.abs(diff_grid)))
    plt.title(f"Difference (Plane 0)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_dir / f"grid_comparison_{test_fname}.png")
    plt.close()

    # Thresholds
    # Adjust these based on expected precision (e.g., float32 vs float64, non-determinism)
    RMS_THRESHOLD = 1e-2 
    
    if rms_diff > RMS_THRESHOLD:
        logger.error(f"RMS difference {rms_diff} exceeds threshold {RMS_THRESHOLD}")
        return False

    logger.info("Simulation consistency check passed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--detector_props", default="src/larndsim/detector_properties/module0.yaml")
    parser.add_argument("--pixel_layouts", default="src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
    args = parser.parse_args()

    success = check_simulation_consistency(args.ref_file, args.test_file, args)
    if not success:
        sys.exit(1)
