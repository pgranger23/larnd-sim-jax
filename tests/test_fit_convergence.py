import pickle
import numpy as np
import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_fit_convergence(input_file, loss_threshold=None, strict=False):
    logger.info(f"Checking convergence for {input_file}")
    
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} does not exist.")
        return False

    with open(input_file, 'rb') as f:
        results = pickle.load(f)

    # 1. Check for NaNs in parameters
    fit_params = [key.replace('_target', '') for key in results.keys() if '_target' in key]
    for par in fit_params:
        values = results.get(f'{par}_iter', [])
        if np.any(np.isnan(values)):
            logger.error(f"Parameter {par} contains NaNs.")
            return False
        if len(values) > 0 and np.isinf(values[-1]):
             logger.error(f"Parameter {par} diverged to Infinity.")
             return False

    # 2. Check Loss Decrease
    losses = results.get('losses_iter', [])
    if len(losses) < 2:
        logger.warning("Not enough iterations to check loss decrease.")
        return True # Can't fail if we didn't run enough

    initial_loss = np.mean(losses[:5]) # Average of first few
    final_loss = np.mean(losses[-5:])   # Average of last few
    
    logger.info(f"Initial Loss: {initial_loss:.4e}, Final Loss: {final_loss:.4e}")

    if final_loss > initial_loss and strict:
        logger.error("Loss increased! Fit diverged.")
        return False
    
    if loss_threshold is not None:
        if final_loss > loss_threshold:
            logger.error(f"Final loss {final_loss} is above threshold {loss_threshold}")
            return False

    logger.info("Fit convergence checks passed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to fit result pickle")
    parser.add_argument("--loss_threshold", type=float, default=None, help="Maximum acceptable final loss")
    parser.add_argument("--strict", action="store_true", help="Fail if loss increases")
    args = parser.parse_args()

    success = check_fit_convergence(args.input_file, args.loss_threshold, args.strict)
    
    if not success:
        sys.exit(1)
