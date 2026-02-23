import h5py
from larndsim.consts_jax import build_params_class, load_detector_properties
from larndsim.fee_jax import get_adc_values
import jax
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--threshold', type=float, default=5e3, help='Discriminator threshold in e-')
    parser.add_argument('-n', type=int, default=10000, help='Number of iterations to run')
    parser.add_argument('--no-noise', action='store_true', help='Disable noise addition')
    args = parser.parse_args()

    Params = build_params_class([])
    ref_params = load_detector_properties(Params,
                                        "/sdf/group/neutrino/pgranger/larnd-sim-jax-bak/src/larndsim/detector_properties/module0.yaml",
                                        '/sdf/group/neutrino/pgranger/larnd-sim-jax-bak/src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml')

    ref_params = ref_params.replace(DISCRIMINATION_THRESHOLD=args.threshold)

    if args.no_noise:
        ref_params = ref_params.replace(RESET_NOISE_CHARGE=0, UNCORRELATED_NOISE_CHARGE=0)

    master_key = jax.random.key(0)
    rngkey1, rngkey2 = jax.random.split(master_key)

    with h5py.File("/sdf/group/neutrino/pgranger/larnd-sim-jax-bak/old_wfs.h5", 'r') as f:
        wfs = f['wfs'][:]
        unique_pixels = f['unique_pixels'][:]

    integral_all = jax.numpy.zeros((args.n, wfs.shape[0], 10))
    ticks_all = jax.numpy.zeros((args.n, wfs.shape[0], 10), dtype=jax.numpy.int32)

    for i in tqdm(range(args.n)):
        rngkey2, _ = jax.random.split(rngkey2)
        integral, ticks = get_adc_values(ref_params, wfs[:, 1:], rngkey2)
        integral_all = integral_all.at[i].set(integral)
        ticks_all = ticks_all.at[i].set(ticks)

    if args.no_noise:
        ofname = f"adc_output_no_noise_thresh{int(args.threshold)}.h5"
    else:
        ofname = f"adc_output_thresh{int(args.threshold)}.h5"

    with h5py.File(ofname, 'w') as f:
        f.create_dataset("results", data=integral_all)
        f.create_dataset("results_ticks", data=ticks_all)

    print("DONE")