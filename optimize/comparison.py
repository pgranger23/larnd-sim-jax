import matplotlib.pyplot as plt
import h5py
import numpy as np
from larndsim.consts_jax import build_params_class, load_detector_properties
from larndsim.sim_jax import id2pixel, get_pixel_coordinates
from glob import glob
plt.rcParams['font.size'] = 15

import argparse

def asciihist(it, bins=10, minmax=None, str_tag='',
              scale_output=30, generate_only=False, print_function=print):
    """Create an ASCII histogram from an interable of numbers.

    Author: Boris Gorelik boris@gorelik.net. based on  http://econpy.googlecode.com/svn/trunk/pytrix/pytrix.py
    License: MIT
    """
    ret = []
    itarray = np.asanyarray(it)
    if minmax == 'auto':
        minmax = np.percentile(it, [5, 95])
        if minmax[0] == minmax[1]:
            # for very ugly distributions
            minmax = None
    if minmax is not None:
        # discard values that are outside minmax range
        mn = minmax[0]
        mx = minmax[1]
        itarray = itarray[itarray >= mn]
        itarray = itarray[itarray <= mx]
    if itarray.size:
        total = len(itarray)
        counts, cutoffs = np.histogram(itarray, bins=bins)
        cutoffs = cutoffs[1:]
        if str_tag:
            str_tag = '%s ' % str_tag
        else:
            str_tag = ''
        if scale_output is not None:
            scaled_counts = counts.astype(float) / counts.sum() * scale_output
        else:
            scaled_counts = counts

        if minmax is not None:
            ret.append('Trimmed to range (%s - %s)' % (str(minmax[0]), str(minmax[1])))
        for cutoff, original_count, scaled_count in zip(cutoffs, counts, scaled_counts):
            ret.append("{:s}{:>8.2f} |{:<7,d} | {:s}".format(
                str_tag,
                cutoff,
                original_count,
                "*" * int(scaled_count))
                       )
        ret.append(
            "{:s}{:s} |{:s} | {:s}".format(
                str_tag,
                '-' * 8,
                '-' * 7,
                '-' * 7
            )
        )
        ret.append(
            "{:s}{:>8s} |{:<7,d}".format(
                str_tag,
                'N=',
                total
            )
        )
    else:
        ret = []
    if not generate_only:
        for line in ret:
            print_function(line)
    ret = '\n'.join(ret)
    return ret


def load_results(fname):
    with h5py.File(fname, 'r') as f:
        pixels = np.array(f['pixels'])
        adc = np.array(f['adc'])
        ticks = np.array(f['ticks'])
    return pixels, adc, ticks

def make_grids(pixels, adcs, ref_params):
    xpitch, ypitch, plane, eid = id2pixel(ref_params, pixels)
    coords = get_pixel_coordinates(ref_params, xpitch, ypitch, np.zeros(pixels.shape[0]))
    grids = []
    for p in [0, 1]:
        xbins = np.linspace(ref_params.tpc_borders[p][0][0], ref_params.tpc_borders[p][0][1], ref_params.n_pixels_x + 1)
        ybins = np.linspace(ref_params.tpc_borders[p][1][0], ref_params.tpc_borders[p][1][1], ref_params.n_pixels_y + 1)
        grid, xbins, ybins = np.histogram2d(coords[p == plane][:, 0], coords[p == plane][:, 1], bins=(xbins, ybins), weights=np.sum(adcs - adcs[0, -1], axis=1)[p == plane])
        grids.append(grid)
    return grids

def compare(config):
    all_diffs = []
    Params = build_params_class([])
    ref_params = load_detector_properties(Params, config.detector_props, config.pixel_layouts)
    nb_not_both_activated = 0

    for i in range(config.n_files):
        pixels_ref, adcs_ref, _ = load_results(f"{config.ref_output}{i}.h5")
        grids_ref = make_grids(pixels_ref, adcs_ref, ref_params)

        pixels_new, adcs_new, _ = load_results(f"{config.output}{i}.h5")
        grids_new = make_grids(pixels_new, adcs_new, ref_params)

        
        for grid_ref, grid_new in zip(grids_ref, grids_new):
            mask = (grid_ref > 0) | (grid_new > 0)
            diff = grid_new - grid_ref
            all_diffs.append(diff[mask])
            nb_not_both_activated += np.sum((grid_ref > 0) != (grid_new > 0))
    all_diffs = np.concatenate(all_diffs)
    asciihist(all_diffs, minmax=None, bins=np.linspace(-10.5, 10.5, 22), str_tag='Distrib');
    print("Average deviation per active pixel:", np.mean(all_diffs))
    print("RMS deviation per active pixel:", np.std(all_diffs))
    print("Max deviation per active pixel:", np.max(all_diffs))
    print("Min deviation per active pixel:", np.min(all_diffs))
    print("Number of active pixels:", len(all_diffs))
    print("Number of pixels that are not both activated:", nb_not_both_activated)
    print("Fraction of pixels that are not both activated:", nb_not_both_activated / len(all_diffs))
    print("Fraction of pixels that are not both activated (in %):", nb_not_both_activated / len(all_diffs) * 100)


    # plt.hist(all_diffs, bins=21, histtype='step', range=(-10, 10), density=True, label="Jax param. vs ref")
    # plt.xlabel("ADC difference")
    # plt.ylabel("Density")
    # plt.tight_layout()
    # plt.savefig("output/diff.png")

    # plt.figure()
    # plt.pcolor(grids_ref[0])
    # plt.colorbar()
    # plt.figure()
    # plt.pcolor(grids_new[0])
    # plt.colorbar()
    # plt.figure()
    # plt.pcolor(grids_new[0] - grids_ref[0])
    # plt.colorbar()
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_output", dest="ref_output", help="Ref data file", required=True)
    parser.add_argument('--output', dest='output', help='Output file', required=True)
    parser.add_argument("--detector_props", dest="detector_props",
                        default="larndsim/detector_properties/module0.yaml",
                        help="Path to detector properties YAML file")
    parser.add_argument("--pixel_layouts", dest="pixel_layouts",
                        default="larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml",
                        help="Path to pixel layouts YAML file")
    parser.add_argument("--n_files", dest="n_files", type=int, default=1, help="Number of files to compare")
    args = parser.parse_args()
    compare(args)
