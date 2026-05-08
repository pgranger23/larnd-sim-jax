import streamlit as st
import sys, os
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.losses_jax import adc2charge
from optimize.strategies import LUTSimulation, LUTProbabilisticSimulation
from optimize.dataio import TracksDataset, DataLoader
from optimize.ranges import ranges

st.set_page_config(page_title="LAr-ND Event Dashboard", layout="wide")

st.title("Interactive Event Display & Simulation Debugger")

# --- Sidebar Configuration ---
st.sidebar.header("Data Configuration")

INPUT_FILE = st.sidebar.text_input("Input H5 File", "/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm.h5")
LUT_FILE = st.sidebar.text_input("LUT File (NPY)", "src/larndsim/detector_properties/response_44.npy")
DET_PROPS = st.sidebar.text_input("Detector Props (YAML)", "src/larndsim/detector_properties/module0.yaml")
PIXEL_LAYOUTS = st.sidebar.text_input("Pixel Layouts (YAML)", "src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")

RELEVANT_PARAMS = ['Ab', 'kb', 'lifetime', 'tran_diff', 'long_diff', 'eField', 'shift_x', 'shift_y', 'shift_z']

st.sidebar.header("Simulation Settings")
FORCE_CPU = st.sidebar.checkbox("Force CPU Execution", value=False)
if FORCE_CPU:
    jax.config.update('jax_platform_name', 'cpu')

ELECTRON_SAMPLING_RESOLUTION = st.sidebar.number_input("Electron Sampling Res (cm)", value=0.01, step=0.01)
NUMBER_PIX_NEIGHBORS = st.sidebar.number_input("Number of Pixel Neighbors", value=4)
SIGNAL_LENGTH = st.sidebar.number_input("Signal Length (ticks)", value=250)
MODE = st.sidebar.selectbox("Simulation Mode", ["Deterministic (Stochastic)", "Probabilistic", "Load from File"])
PRE_SIM_FILE = None
if MODE == "Load from File":
    PRE_SIM_FILE = st.sidebar.text_input("Pre-simulated H5 File", "output.h5")

st.sidebar.header("Physics Parameters")
current_values = {}
for p in RELEVANT_PARAMS:
    p_range = ranges[p]
    current_values[p] = st.sidebar.slider(f"{p}", float(p_range['min']), float(p_range['max']), float(p_range['nom']), format="%.2e" if p_range['nom'] < 0.01 else "%.3f")

# --- Resource Caching ---
@st.cache_resource
def get_dataset(filename, res):
    try:
        dataset = TracksDataset(filename=filename, nevents=-1, max_nbatch=-1, print_input=False, 
                                max_batch_len=200, electron_sampling_resolution=res)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def get_detector_stuff(det_props, pixel_layouts, lut_file, res, neighbors, sig_len):
    ParamsClass = build_params_class(RELEVANT_PARAMS)
    ref_params = load_detector_properties(ParamsClass, det_props, pixel_layouts)
    ref_params = ref_params.replace(
        electron_sampling_resolution=res,
        number_pix_neighbors=neighbors,
        signal_length=sig_len,
        time_window=sig_len,
    )
    response, ref_params = load_lut(lut_file, ref_params)
    return response, ref_params

# --- Simulation Caching ---
@st.cache_data
def run_live_simulation(_response, _ref_params, _tracks_jax, _track_fields, params_values, mode, event_idx, input_file):
    current_params = _ref_params.replace(**params_values)
    
    if mode == "Deterministic (Stochastic)":
        strategy = LUTSimulation(_response)
        output = strategy.predict(current_params, _tracks_jax, _track_fields, rngkey=42)
    else:
        strategy = LUTProbabilisticSimulation(_response)
        output = strategy.predict(current_params, _tracks_jax, _track_fields, rngkey=42)
    
    # Convert JAX arrays to numpy for caching and plotly
    return jax.device_get(output)

@st.cache_data
def load_sim_from_file(filename, event_idx, global_event_ids):
    import h5py
    output = {}
    if not filename or not os.path.exists(filename):
        return None
        
    with h5py.File(filename, 'r') as f:
        all_adcs, all_x, all_y, all_z, all_ticks, all_events, all_pixels = [], [], [], [], [], [], []
        
        batch_key = f"batch_{event_idx}"
        if batch_key in f:
            bg = f[batch_key]
            for geid in global_event_ids:
                evt_key = f"event_{geid}"
                if evt_key in bg:
                    eg = bg[evt_key]
                    if 'adc' in eg: all_adcs.append(np.array(eg['adc']))
                    if 'pix_x' in eg: all_x.append(np.array(eg['pix_x']))
                    if 'pix_y' in eg: all_y.append(np.array(eg['pix_y']))
                    if 'pix_z' in eg: all_z.append(np.array(eg['pix_z']))
                    if 'ticks' in eg: all_ticks.append(np.array(eg['ticks']))
                    if 'eventID' in eg: all_events.append(np.array(eg['eventID']))
                    if 'pixels' in eg: all_pixels.append(np.array(eg['pixels']))
                    if 'wfs' in eg and 'wfs' not in output:
                        output['wfs'] = np.array(eg['wfs'])
        
        if all_adcs:
            output['adcs'] = np.concatenate(all_adcs)
            output['pixel_x'] = np.concatenate(all_x)
            output['pixel_y'] = np.concatenate(all_y)
            output['pixel_z'] = np.concatenate(all_z)
            output['ticks'] = np.concatenate(all_ticks)
            output['event'] = np.concatenate(all_events)
            if all_pixels:
                output['unique_pixels'] = np.unique(np.concatenate(all_pixels))
            else:
                output['unique_pixels'] = np.unique(output['pixel_x'] * 10000 + output['pixel_y'])
            return output
    return None

# --- Load Resources ---
dataset = get_dataset(INPUT_FILE, ELECTRON_SAMPLING_RESOLUTION)
if dataset:
    response, ref_params = get_detector_stuff(DET_PROPS, PIXEL_LAYOUTS, LUT_FILE, 
                                             ELECTRON_SAMPLING_RESOLUTION, NUMBER_PIX_NEIGHBORS, SIGNAL_LENGTH)

    st.sidebar.header("Event Selection")
    num_events = len(dataset)
    event_idx = st.sidebar.number_input(f"Batch Index (0 to {num_events-1})", 0, num_events-1, 0)

    # --- Run Simulation ---
    tracks_batch = dataset[event_idx]
    track_fields = dataset.get_track_fields()
    
    # We want to keep the original tracks for plotting truth
    valid_tracks_mask = tracks_batch[:, track_fields.index('eventID')] >= 0
    valid_tracks = tracks_batch[valid_tracks_mask]
    
    tracks_jax = jax.device_put(tracks_batch.reshape(-1, len(track_fields)))

    output = None
    with st.spinner("Running/Loading Simulation..."):
        if MODE == "Load from File":
            global_event_ids = dataset.get_batch_global_event_ids(event_idx)
            output = load_sim_from_file(PRE_SIM_FILE, event_idx, global_event_ids)
            if output is None:
                st.warning(f"No data found for Batch {event_idx} in {PRE_SIM_FILE}")
        else:
            output = run_live_simulation(response, ref_params, tracks_jax, track_fields, current_values, MODE, event_idx, INPUT_FILE)

    if output:
        # --- Visualization ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("3D Event Display")
            
            fig_3d = go.Figure()

            # Truth tracks
            x_start = valid_tracks[:, track_fields.index('x_start')]
            y_start = valid_tracks[:, track_fields.index('y_start')]
            z_start = valid_tracks[:, track_fields.index('z_start')]
            x_end = valid_tracks[:, track_fields.index('x_end')]
            y_end = valid_tracks[:, track_fields.index('y_end')]
            z_end = valid_tracks[:, track_fields.index('z_end')]

            t_x = np.column_stack([x_start, x_end, np.full_like(x_start, np.nan)]).flatten()
            t_y = np.column_stack([y_start, y_end, np.full_like(y_start, np.nan)]).flatten()
            t_z = np.column_stack([z_start, z_end, np.full_like(z_start, np.nan)]).flatten()

            # Bright color for tracks on dark background
            fig_3d.add_trace(go.Scatter3d(x=t_x, y=t_y, z=t_z, mode='lines', name='Truth Tracks', 
                                         line=dict(color='cyan', width=4)))

            # Simulated Hits
            if 'adcs' in output:
                fig_3d.add_trace(go.Scatter3d(
                    x=output['pixel_x'], y=output['pixel_y'], z=output['pixel_z'],
                    mode='markers', name='Simulated Hits',
                    marker=dict(size=2.5, color=output['adcs'], colorscale='Hot', showscale=True, opacity=0.8),
                    hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>ADC: %{marker.color}<extra></extra>"
                ))
            elif 'hit_prob' in output:
                from larndsim.detsim_jax import id2pixel, get_hit_z
                from larndsim.fee_jax import get_average_hit_values
                
                ticks_prob = output['hit_prob'] # This is log-probability
                adcs_distrib = output['adcs_distrib']
                unique_pixels = output['unique_pixels']
                
                # These helpers expect params object
                current_params = ref_params.replace(**current_values)
                expected_ticks, expected_adcs, hit_prob = get_average_hit_values(ticks_prob, adcs_distrib)
                Npix, Nhits = expected_ticks.shape
                
                _, _, pixel_plane, _ = id2pixel(current_params, unique_pixels)
                pred_z = get_hit_z(current_params, expected_ticks, pixel_plane[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32))
                
                mask = np.array(hit_prob > 0.01)
                if np.any(mask):
                    plot_x = np.array(output['pixel_x'][:, None] * np.ones((Npix, Nhits)))[mask]
                    plot_y = np.array(output['pixel_y'][:, None] * np.ones((Npix, Nhits)))[mask]
                    plot_z = np.array(pred_z)[mask]
                    plot_prob = np.array(hit_prob)[mask]
                    plot_adc = np.array(expected_adcs)[mask]

                    fig_3d.add_trace(go.Scatter3d(
                        x=plot_x, y=plot_y, z=plot_z,
                        mode='markers', name='Simulated Pseudo-Hits',
                        marker=dict(
                            size=2 + plot_prob * 10, # Increased scaling for visibility
                            color=plot_adc, 
                            colorscale='YlOrRd', 
                            showscale=True, 
                            opacity=0.7 
                        ),
                        customdata=plot_prob,
                        hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Prob: %{customdata:.4f}<br>Exp ADC: %{marker.color:.1f}<extra></extra>"
                    ))
                else:
                    st.info("No pseudo-hits passed the probability threshold (> 0.01).")

            # Detector Boundaries (simplified)
            for drift_sign in [-1, 1]:
                for side in [-1, 1]:
                    box_x = [side*30, side*30, side*30, side*30, side*30]
                    box_y = [-60, 60, 60, -60, -60]
                    box_z = [0, 0, drift_sign*30, drift_sign*30, 0]
                    fig_3d.add_trace(go.Scatter3d(x=box_x, y=box_y, z=box_z, mode='lines', 
                                                 line=dict(color='white', width=1, dash='dash'), 
                                                 showlegend=False, hoverinfo='skip'))

            fig_3d.update_layout(template='plotly_dark',
                                 scene=dict(xaxis_title='X (cm)', yaxis_title='Y (cm)', zaxis_title='Z (cm)'),
                                 height=700, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, use_container_width=True)

        with col2:
            st.subheader("Waveform Analysis")
            
            if 'wfs' in output and 'unique_pixels' in output:
                unique_pixels = output['unique_pixels']
                max_val_per_wf = np.max(output['wfs'], axis=1)
                
                # Filter out garbage -1 pixels
                valid_pix_mask = unique_pixels >= 0
                v_pixels = unique_pixels[valid_pix_mask]
                v_max_vals = max_val_per_wf[valid_pix_mask]
                
                if len(v_pixels) > 0:
                    best_wf_idx_in_valid = np.argmax(v_max_vals)
                    
                    pixel_options = [f"Pixel {pid} (Max ADC: {mv:.1f})" for pid, mv in zip(v_pixels, v_max_vals)]
                    selected_pix_label = st.selectbox("Select Pixel to Inspect", pixel_options, index=int(best_wf_idx_in_valid))
                    
                    # Map back to original index in output['wfs']
                    selected_pid = int(selected_pix_label.split()[1])
                    selected_idx = np.where(unique_pixels == selected_pid)[0][0]
                    
                    wf = output['wfs'][selected_idx]
                    fig_wf = go.Figure()
                    fig_wf.add_trace(go.Scatter(x=np.arange(len(wf)), y=wf, mode='lines', name='Waveform', line=dict(color='orange')))
                    fig_wf.update_layout(template='plotly_dark', xaxis_title="Tick", yaxis_title="Current/ADC", height=400)
                    st.plotly_chart(fig_wf, use_container_width=True)

                    if MODE == "Probabilistic" and 'hit_prob' in output:
                        st.subheader("Probabilistic Distributions")
                        # Exponentiate log-probability
                        prob_dist = np.exp(output['hit_prob'][selected_idx, 0, :])
                        fig_prob = go.Figure()
                        fig_prob.add_trace(go.Scatter(x=np.arange(len(prob_dist)), y=prob_dist, mode='lines', name='Hit Probability', line=dict(color='red')))
                        fig_prob.update_layout(template='plotly_dark', xaxis_title="Tick", yaxis_title="Probability", height=300)
                        st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.info("No valid active pixels in this event.")

        # --- Stats ---
        st.markdown("---")
        st.subheader("Simulation Stats")
        c1, c2, c3 = st.columns(3)
        c1.metric("Num Truth Tracks", len(valid_tracks))
        if 'adcs' in output:
            c2.metric("Num Hits", len(output['adcs']))
        if 'unique_pixels' in output:
            c3.metric("Num Unique Pixels", len(output['unique_pixels'][output['unique_pixels'] >= 0]))

else:
    st.warning("Please verify the input file path.")
