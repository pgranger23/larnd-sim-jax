import jax
import jax.numpy as jnp
from optimize.strategies import MarkovLossStrategy

def main():
    # Create dummy data
    Npix = 2
    Nticks = 10
    
    # Prediction (Markov arrays)
    prediction = {
        'log_p1': jnp.zeros((Npix, Nticks)),
        'log_T': jnp.zeros((Npix, Nticks, Nticks)),
        'expected_Q': jnp.ones((Npix, Nticks, Nticks)) * 1000.0,
        'log_p_none': jnp.full((Npix, Nticks), -10.0),
        'Q1': jnp.ones((Npix, Nticks)) * 1000.0,
        'log_p_none_at_zero': jnp.full((Npix,), -10.0),
        'unique_pixels': jnp.array([100, 200])
    }
    
    # Target
    target = {
        'hit_pixels': jnp.array([100, 100, 200]),
        'ticks': jnp.array([2, 5, 8]),
        'adcs': jnp.array([10.0, 15.0, 12.0])
    }
    
    class DummyParams:
        GAIN = 0.004
        V_PEDESTAL = 50.0
        V_CM = 20.0
        ADC_COUNTS = 256
        V_REF = 100.0
        fee_paths_scaling = 1
        RESET_NOISE_CHARGE = 10.0
        DISCRIMINATION_THRESHOLD = 5.0
        CLOCK_CYCLE = 0.1
        ADC_HOLD_DELAY = 1.0
        t_sampling = 0.1
        roi_threshold = 2.0
    
    params = DummyParams()
    
    loss_strategy = MarkovLossStrategy()
    
    # We can JIT the compute method
    @jax.jit
    def test_compute(pred, targ):
        return loss_strategy.compute(params, pred, targ)
        
    loss, aux = test_compute(prediction, target)
    print("Loss:", loss)
    print("Aux:", aux)

if __name__ == "__main__":
    main()
