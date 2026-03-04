import streamlit as st
import glob
import pickle
import time
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors

# -- Configuration --
st.set_page_config(page_title="Fit & Scan Monitor", layout="wide")

# Standard color palette for coordinating lines
COLORS = plotly.colors.qualitative.Plotly

# -- Global Styling Dictionary --
COMMON_LAYOUT = dict(
    height=400,
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=-0.2, 
        xanchor="center", 
        x=0.5,
        font=dict(size=10)
    ),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

AXIS_STYLE = dict(
    showgrid=True, 
    gridcolor='rgba(128, 128, 128, 0.2)', 
    zerolinecolor='rgba(128, 128, 128, 0.5)'
)

def get_available_folders():
    all_pkls = glob.glob("**/*.pkl", recursive=True)
    folders = list(set([os.path.dirname(p) for p in all_pkls]))
    folders = [f if f != "" else "." for f in folders]
    try:
        folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    except OSError:
        pass
    return folders

def safe_load(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, FileNotFoundError, OSError):
        return None

def smooth_data(data, window):
    if window <= 1 or len(data) == 0:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values

def compute_folder_signature(files):
    signature = []
    for f in files:
        try:
            stat = os.stat(f)
            signature.append((f, stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((f, 0, 0))
    return tuple(sorted(signature))

def prepare_log_values(values):
    arr = np.asarray(values, dtype=float)
    positive = arr > 0
    if not np.any(positive):
        return arr, False
    min_pos = np.min(arr[positive])
    floor = min_pos * 0.1
    return np.where(arr <= 0, floor, arr), True

def to_xy_series(values):
    y_arr = np.asarray(values, dtype=float)
    x_arr = np.arange(len(y_arr), dtype=float)
    return x_arr, y_arr

# -- Scan Extraction Logic --
def extract_param_from_filename(fname, local_params):
    basename = os.path.basename(fname)
    if not basename.startswith('history_'):
        return None, None
    parts = basename.split('_')
    for i in range(1, len(parts)):
        potential_param = '_'.join(parts[1:i+1])
        if potential_param in local_params:
            return potential_param, None 
    return None, None

def extract_scan_data(results, config, fname):
    local_params = [key.replace('_grad', '') for key in results.keys() if '_grad' in key]
    if not local_params:
        return None, None, None, None, "Unknown", None

    nparams = len(local_params)
    nb_iter = config.iterations
    total_points = len(results.get("losses_iter", []))
    
    param_name, _ = extract_param_from_filename(fname, local_params)
    if param_name is None or param_name not in local_params:
        param_name = local_params[0]
        
    param_idx = local_params.index(param_name)
    n_scans = total_points // nb_iter
    num_batches = (n_scans + nparams - 1) // nparams
    
    param_values_list, grad_list, loss_list = [], [], []
    aux_data = {}
    
    for b in range(num_batches):
        scan_idx = b * nparams + param_idx
        if scan_idx >= n_scans: break
        
        start_iter = 1 + scan_idx * nb_iter
        end_iter = start_iter + nb_iter
        start_data = scan_idx * nb_iter
        end_data = start_data + nb_iter
        
        param_values_list.append(results[f"{param_name}_iter"][start_iter:end_iter])
        grad_list.append(results[f"{param_name}_grad"][start_data:end_data])
        loss_list.append(results["losses_iter"][start_data:end_data])
        
        if 'aux_iter' in results and len(results['aux_iter']) > 0:
            aux_slice = results['aux_iter'][start_data:end_data]
            if b == 0 and len(aux_slice) > 0:
                first_entry = aux_slice[0] if isinstance(aux_slice[0], dict) else {}
                for key in first_entry.keys(): aux_data[key] = []
            for key in aux_data.keys():
                batch_values = []
                for entry in aux_slice:
                    if isinstance(entry, dict) and key in entry:
                        val = entry[key]
                        batch_values.append(float(val) if isinstance(val, (float, int)) else np.nan)
                    else:
                        batch_values.append(np.nan)
                aux_data[key].append(batch_values)
                
    param_values = np.array(param_values_list)
    gradients = np.array(grad_list)
    losses = np.array(loss_list)
    for key in aux_data.keys(): aux_data[key] = np.array(aux_data[key])
    target = results.get(f"{param_name}_target", [None])[0] if f"{param_name}_target" in results else None
    
    return param_values, gradients, losses, aux_data, param_name, target


# -- HTML Report Generator --
def generate_html_report(figs_to_export, config_data):
    """Stitches Plotly figures and Config into a single responsive HTML file."""
    
    # Format config as an HTML table string
    config_html = ""
    if config_data:
        # Convert to dict if it's an object
        c_dict = vars(config_data) if hasattr(config_data, '__dict__') else config_data
        if isinstance(c_dict, dict):
            rows = "".join([f"<tr><td style='font-weight:bold; padding:4px;'>{k}</td><td style='padding:4px;'>{v}</td></tr>" for k, v in c_dict.items()])
            config_html = f"""
            <div class='config-container'>
                <h2>Configuration</h2>
                <div style='max-height: 200px; overflow-y: auto; border: 1px solid #ddd;'>
                    <table style='width:100%; border-collapse: collapse; font-family: monospace; font-size: 12px;'>
                        {rows}
                    </table>
                </div>
            </div>
            """

    html = [
        "<html><head><title>Fit Dashboard Report</title>",
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.1.1/plotly.min.js'></script>",
        "<style>",
        "body { font-family: sans-serif; padding: 20px; background-color: #f9f9f9; }",
        ".grid { display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 20px; }",
        ".config-container { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 15px; margin-bottom: 20px; }",
        ".plot-container { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 15px; width: 100%; min-width: 0; }",
        "h2 { text-align: center; color: #333; margin-top: 0; }",
        "@media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, minmax(280px, 1fr)); } }",
        "@media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }",
        "@media print { .grid { grid-template-columns: 1fr; } .plot-container { box-shadow: none; page-break-inside: avoid; } .config-container { page-break-inside: avoid; } }",
        "</style></head><body>",
        "<h1 style='text-align:center;'>Monitoring Report</h1>",
        config_html,
        "<div class='grid'>"
    ]
    for title, fig in figs_to_export:
        html.append(f"<div class='plot-container'><h2>{title}</h2>")
        html.append(fig.to_html(full_html=False, include_plotlyjs=False))
        html.append("</div>")
    html.append("</div></body></html>")
    return "\n".join(html)


# -- Mode Rendering Functions --
def render_scan_mode(all_data, plot_all, export_list):
    st.header("Gradient Scan Results")

    for filepath, results in all_data.items():
        config = results.get('config')
        p_vals, grads, losses, aux, p_name, target = extract_scan_data(results, config, filepath)
        
        if p_vals is None or len(p_vals) == 0:
            continue

        st.subheader(f"Scan: {p_name} ({os.path.basename(filepath)})")
        cols = st.columns(3)

        with cols[0]:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            p_vals_plot = p_vals.T if plot_all else np.nanmean(p_vals, axis=0)
            grads_plot = grads.T if plot_all else np.nanmean(grads, axis=0)
            losses_plot = losses.T if plot_all else np.nanmean(losses, axis=0)
            
            if plot_all and p_vals_plot.ndim > 1:
                for i in range(p_vals_plot.shape[1]):
                    show_leg = (i == 0)
                    fig.add_trace(go.Scatter(x=p_vals_plot[:, i], y=grads_plot[:, i], mode='lines', line=dict(color='#1f77b4', width=1.5), name='Gradient', showlegend=show_leg), secondary_y=False)
                    fig.add_trace(go.Scatter(x=p_vals_plot[:, i], y=losses_plot[:, i], mode='lines', line=dict(color='#2ca02c', width=1.5), name='Loss', showlegend=show_leg), secondary_y=True)
            else:
                fig.add_trace(go.Scatter(x=p_vals_plot, y=grads_plot, mode='lines', line=dict(color='#1f77b4', width=2), name='Gradient'), secondary_y=False)
                fig.add_trace(go.Scatter(x=p_vals_plot, y=losses_plot, mode='lines', line=dict(color='#2ca02c', width=2), name='Loss'), secondary_y=True)
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Zero Grad")
            if target is not None:
                fig.add_vline(x=target, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Target")
                
            fig.update_layout(**COMMON_LAYOUT)
            fig.update_layout(title="Gradient & Loss", hovermode="closest")
            fig.update_xaxes(title_text=p_name, **AXIS_STYLE)
            fig.update_yaxes(title_text="Gradient", title_font=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4"), secondary_y=False, **AXIS_STYLE)
            fig.update_yaxes(title_text="Loss", title_font=dict(color="#2ca02c"), tickfont=dict(color="#2ca02c"), secondary_y=True, **AXIS_STYLE)
            st.plotly_chart(fig, width='stretch')
            export_list.append((f"{p_name} - Gradient & Loss", fig))

        with cols[1]:
            fig = go.Figure()
            if aux:
                for idx, (key, values) in enumerate(aux.items()):
                    c = COLORS[idx % len(COLORS)]
                    if plot_all and values.ndim > 1:
                        v_plot = values.T
                        for i in range(v_plot.shape[1]):
                            fig.add_trace(go.Scatter(x=p_vals_plot[:, i] if plot_all else p_vals_plot, y=v_plot[:, i], mode='lines', line=dict(color=c, width=1.5), name=key, showlegend=(i==0)))
                    else:
                        v_plot = np.nanmean(values, axis=0)
                        fig.add_trace(go.Scatter(x=p_vals_plot, y=v_plot, mode='lines', line=dict(color=c, width=2), name=key))
                
                if target is not None:
                    fig.add_vline(x=target, line_dash="dash", line_color="red", opacity=0.5)
                
                fig.update_layout(**COMMON_LAYOUT)
                fig.update_layout(title="Sub-loss terms", xaxis_title=p_name, yaxis_title="Sub-loss values", hovermode="closest")
                fig.update_xaxes(**AXIS_STYLE)
                fig.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig, width='stretch')
                export_list.append((f"{p_name} - Sub-losses", fig))
            else:
                st.info("No auxiliary data")

        with cols[2]:
            fig = go.Figure()
            if 'step_time' in results:
                fig.add_trace(go.Scatter(y=results['step_time'], mode='lines', name='Time', line=dict(color=COLORS[0], width=2)))
                fig.update_layout(**COMMON_LAYOUT)
                fig.update_layout(title="Time per iteration", xaxis_title="Iteration", yaxis_title="Time (s)")
                fig.update_xaxes(**AXIS_STYLE)
                fig.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig, width='stretch')
                export_list.append((f"{p_name} - Time", fig))
        
        st.markdown("---")

def render_optimization_mode(all_data, smoothing_window, export_list):
    st.header("Optimization Monitoring")
    
    params_set = set()
    for data in all_data.values():
        params_set.update([k.replace('_target', '') for k in data.keys() if '_target' in k])
    params = sorted(list(params_set))

    st.subheader("Global Metrics")
    global_cols = st.columns(3)
    
    with global_cols[0]:
        fig = go.Figure()
        has_positive = False
        for idx, (fp, d) in enumerate(all_data.items()):
            name = os.path.splitext(os.path.basename(fp))[0]
            if 'losses_iter' in d:
                y_smooth = smooth_data(d['losses_iter'], smoothing_window)
                y_smooth, has_pos = prepare_log_values(y_smooth)
                has_positive = has_positive or has_pos
                x_arr, y_arr = to_xy_series(y_smooth)
                if len(y_arr) > 0:
                    fig.add_trace(go.Scatter(x=x_arr, y=y_arr, mode='lines', name=name, line=dict(color=COLORS[idx % len(COLORS)], width=2)))
        
        fig.update_layout(**COMMON_LAYOUT)
        fig.update_layout(title="Loss Evolution", xaxis_title="Iteration", yaxis_title="Loss")
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(type="log" if has_positive else "linear", **AXIS_STYLE)
        st.plotly_chart(fig, width='stretch')
        export_list.append(("Global - Loss Evolution", fig))

    with global_cols[1]:
        fig = go.Figure()
        for idx, (fp, d) in enumerate(all_data.items()):
            name = os.path.splitext(os.path.basename(fp))[0]
            if 'step_time' in d:
                y_smooth = smooth_data(d['step_time'], smoothing_window)
                fig.add_trace(go.Scatter(y=y_smooth, mode='lines', name=name, line=dict(color=COLORS[idx % len(COLORS)], width=2)))
        
        fig.update_layout(**COMMON_LAYOUT)
        fig.update_layout(title="Step Time vs Iteration", xaxis_title="Iteration", yaxis_title="Time (s)")
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, width='stretch')
        export_list.append(("Global - Step Time", fig))

    with global_cols[2]:
        fig = go.Figure()
        has_time = False
        for idx, (fp, d) in enumerate(all_data.items()):
            name = os.path.splitext(os.path.basename(fp))[0]
            if 'step_time' in d and len(d['step_time']) > 0:
                has_time = True
                fig.add_trace(go.Histogram(x=d['step_time'], name=name, opacity=0.7, marker_color=COLORS[idx % len(COLORS)]))
        if has_time:
            fig.update_layout(**COMMON_LAYOUT)
            fig.update_layout(title="Step Time Distribution", barmode='overlay', xaxis_title="Time (s)", yaxis_title="Frequency", hovermode="closest")
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, width='stretch')
            export_list.append(("Global - Time Distribution", fig))
        else:
            st.info("No step_time data.")

    st.markdown("---")

    for par in params:
        st.subheader(f"Parameter: {par}")
        par_cols = st.columns(3)
        
        with par_cols[0]:
            fig = go.Figure()
            for idx, (fp, d) in enumerate(all_data.items()):
                name = os.path.splitext(os.path.basename(fp))[0]
                c = COLORS[idx % len(COLORS)]
                if f'{par}_iter' in d:
                    y_smooth = smooth_data(d[f'{par}_iter'], smoothing_window)
                    fig.add_trace(go.Scatter(y=y_smooth, mode='lines', name=name, line=dict(color=c, width=2)))
                    
                    target = d.get(f'{par}_target')
                    if target is not None:
                        t_val = target[0] if isinstance(target, (list, np.ndarray)) else target
                        fig.add_hline(y=t_val, line_dash="dash", line_color=c, opacity=0.4)
                        
            fig.update_layout(**COMMON_LAYOUT)
            fig.update_layout(title=f"{par} Evolution", xaxis_title="Iteration", yaxis_title=par)
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, width='stretch')
            export_list.append((f"{par} - Evolution", fig))
            
        with par_cols[1]:
            fig = go.Figure()
            has_positive = False
            for idx, (fp, d) in enumerate(all_data.items()):
                name = os.path.splitext(os.path.basename(fp))[0]
                c = COLORS[idx % len(COLORS)]
                if f'{par}_grad' in d:
                    grads = np.abs(d[f'{par}_grad'])
                    y_smooth = smooth_data(grads, smoothing_window)
                    y_smooth, has_pos = prepare_log_values(y_smooth)
                    has_positive = has_positive or has_pos
                    x_arr, y_arr = to_xy_series(y_smooth)
                    if len(y_arr) > 0:
                        fig.add_trace(go.Scatter(x=x_arr, y=y_arr, mode='lines', name=name, line=dict(color=c, width=2)))
                    
            fig.update_layout(**COMMON_LAYOUT)
            fig.update_layout(title="Absolute Gradient", xaxis_title="Iteration", yaxis_title="|Gradient|")
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(type="log" if has_positive else "linear", **AXIS_STYLE)
            st.plotly_chart(fig, width='stretch')
            export_list.append((f"{par} - Gradient", fig))

        with par_cols[2]:
            fig = go.Figure()
            has_phase = False
            for idx, (fp, d) in enumerate(all_data.items()):
                name = os.path.splitext(os.path.basename(fp))[0]
                c = COLORS[idx % len(COLORS)]
                if f'{par}_iter' in d and f'{par}_grad' in d:
                    min_len = min(len(d[f'{par}_iter']), len(d[f'{par}_grad']))
                    if min_len > 0:
                        has_phase = True
                        val_s = smooth_data(d[f'{par}_iter'][:min_len], smoothing_window)
                        grad_s = smooth_data(d[f'{par}_grad'][:min_len], smoothing_window)
                        
                        fig.add_trace(go.Scatter(x=val_s, y=grad_s, mode='lines', name=name, line=dict(color=c, width=2), opacity=0.8))
                        fig.add_trace(go.Scatter(x=[val_s[0]], y=[grad_s[0]], mode='markers', marker=dict(color=c, symbol='circle', size=10, opacity=0.4, line=dict(width=1, color=c)), showlegend=False, hoverinfo='skip'))
                        fig.add_trace(go.Scatter(x=[val_s[-1]], y=[grad_s[-1]], mode='markers', marker=dict(color=c, symbol='x', size=10, line=dict(width=2, color=c)), showlegend=False, hoverinfo='skip'))
                        
                        target = d.get(f'{par}_target')
                        if target is not None:
                            t_val = target[0] if isinstance(target, (list, np.ndarray)) else target
                            fig.add_vline(x=t_val, line_dash="dash", line_color=c, opacity=0.4)

            if has_phase:
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.update_layout(**COMMON_LAYOUT)
                fig.update_layout(title="Phase Plot", xaxis_title="Parameter Value", yaxis_title="Gradient", hovermode="closest")
                fig.update_xaxes(**AXIS_STYLE)
                fig.update_yaxes(**AXIS_STYLE)
                st.plotly_chart(fig, width='stretch')
                export_list.append((f"{par} - Phase Plot", fig))


# ==========================================
# Main Sidebar & Execution Flow
# ==========================================
st.sidebar.title("Controls")

available_folders = get_available_folders()
if not available_folders:
    st.sidebar.error("No .pkl files found in this directory or subdirectories.")
    st.stop()

if 'current_folder' not in st.session_state:
    st.session_state.current_folder = "." if "." in available_folders else available_folders[0]

if st.session_state.current_folder not in available_folders:
    st.session_state.current_folder = "." if "." in available_folders else available_folders[0]

default_idx = available_folders.index(st.session_state.current_folder)

selected_folder = st.sidebar.selectbox("Select Fit Folder", options=available_folders, index=default_idx)
st.session_state.current_folder = selected_folder
pattern = os.path.join(selected_folder, "*.pkl")

st.sidebar.markdown("---")
refresh_rate = st.sidebar.slider("Refresh Interval (s)", 2, 30, 5)
auto_refresh = st.sidebar.checkbox("Live Auto-Refresh", value=True)
st.sidebar.markdown("---")

files = sorted(glob.glob(pattern))
all_data = {}
figs_to_export = [] # List to collect all figures generated during the loop
config_to_export = None # Placeholder for the config to export

folder_signature = compute_folder_signature(files)
folder_changed = st.session_state.get("cached_folder") != selected_folder
data_changed = st.session_state.get("cached_signature") != folder_signature
needs_reload = folder_changed or data_changed

if not files:
    st.warning(f"No files found matching: {pattern}")
else:
    if needs_reload:
        for f in files:
            data = safe_load(f)
            if data is not None:
                all_data[f] = data

        st.session_state.cached_all_data = all_data
        st.session_state.cached_signature = folder_signature
        st.session_state.cached_folder = selected_folder
    else:
        all_data = st.session_state.get("cached_all_data", {})

    if not all_data:
        st.info("Files found but could not be read (locked?). Retrying...")
    else:
        st.sidebar.markdown(f"**Active Files:** {len(all_data)}")
        
        # --- Config Display Logic ---
        # Grab config from the first valid file
        first_data = next(iter(all_data.values()))
        config_to_export = first_data.get('config', None)

        if config_to_export is not None:
            with st.expander("📂 Run Configuration", expanded=False):
                # Convert argparse Namespace or others to dict for pretty display
                if hasattr(config_to_export, '__dict__'):
                    st.json(vars(config_to_export))
                else:
                    st.json(config_to_export)

        # --- Mode Detection ---
        is_scan = False
        for d in all_data.values():
            if 'config' in d and hasattr(d['config'], 'fit_type'):
                if d['config'].fit_type == 'scan':
                    is_scan = True
                break
        
        if is_scan:
            st.sidebar.subheader("Scan Mode Settings")
            plot_all = st.sidebar.checkbox("Plot all batches", value=True, help="Uncheck to average")
            render_scan_mode(all_data, plot_all, figs_to_export)
        else:
            st.sidebar.subheader("Fit Mode Settings")
            smoothing_window = st.sidebar.slider("Smoothing Window (Iterations)", min_value=1, max_value=100, value=1, step=1)
            render_optimization_mode(all_data, smoothing_window, figs_to_export)

# -- Report Export Button --
if figs_to_export:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export")
    
    # Generate the HTML payload with Config included
    html_report = generate_html_report(figs_to_export, config_to_export)
    
    st.sidebar.download_button(
        label="📥 Download Full Report (HTML)",
        data=html_report,
        file_name=f"fit_report_{os.path.basename(os.path.abspath(selected_folder))}.html",
        mime="text/html",
        help="Downloads an interactive HTML file containing the Configuration and all current plots."
    )

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()