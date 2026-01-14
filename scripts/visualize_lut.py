#!/usr/bin/env python3
"""
Visualize LUT current waveforms to understand how they vary with indices.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/sdf/home/c/cjesus/LArTPC/larnd-sim-jax')

def norm_pdf(x, scale):
    """Gaussian PDF without scipy."""
    return np.exp(-x**2 / (2 * scale**2)) / (scale * np.sqrt(2 * np.pi))

# Output directory
OUTPUT_DIR = "/sdf/home/c/cjesus/LArTPC/dev_figures"

def load_and_build_response_template(lut_file):
    """Load raw LUT and build response_template with diffusion dimension."""

    # Load raw LUT
    data = np.load(lut_file)
    response = data['response']
    metadata = {
        'drift_length': float(data['drift_length']),
        'time_tick': float(data['time_tick']),
        'bin_size': float(data['bin_size'])
    }

    print(f"Raw LUT shape: {response.shape} (Nx, Ny, Nt)")
    print(f"Metadata: {metadata}")

    # Build diffusion dimension (same as load_lut in consts_jax.py)
    long_diff_template = np.linspace(0.001, 10, 100)  # diffusion values in ticks
    long_diff_extent = 20

    # Create Gaussian kernels
    x = np.arange(-long_diff_extent, long_diff_extent + 1, 1)
    gaus = norm_pdf(x, scale=long_diff_template[:, None])
    gaus = gaus / gaus.sum(axis=1, keepdims=True)

    # Convolve each spatial position with each Gaussian
    print("Building response_template with diffusion... (this may take a moment)")
    Nx, Ny, Nt = response.shape
    N_diff = len(long_diff_template)

    response_template = np.zeros((N_diff, Nx, Ny, Nt))
    for i in range(Nx):
        for j in range(Ny):
            for d in range(N_diff):
                response_template[d, i, j, :] = np.convolve(response[i, j, :], gaus[d], mode='same')

    # Set index 0 to raw response (no diffusion)
    response_template[0] = response

    print(f"Response template shape: {response_template.shape} (N_diff, Nx, Ny, Nt)")

    return response_template, long_diff_template, metadata


def plot_varying_position(response_template, metadata, signal_length=200):
    """Plot waveforms varying sub-pixel position (x_bin, y_bin)."""

    t_tick = metadata['time_tick']
    Nt = response_template.shape[-1]

    # Use no diffusion (index 0)
    diff_idx = 0

    # Time axis (last signal_length ticks)
    t_start = Nt - signal_length
    t = np.arange(signal_length) * t_tick

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Vary x_bin at fixed y_bin=2
    ax = axes[0, 0]
    y_bin = 2
    for x_bin in range(5):
        wf = response_template[diff_idx, x_bin, y_bin, t_start:]
        ax.plot(t, wf, label=f'x_bin={x_bin}', linewidth=1.5)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Induced Current (a.u.)')
    ax.set_title(f'Varying x_bin (y_bin={y_bin}, diff_idx={diff_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Vary y_bin at fixed x_bin=2
    ax = axes[0, 1]
    x_bin = 2
    for y_bin in range(5):
        wf = response_template[diff_idx, x_bin, y_bin, t_start:]
        ax.plot(t, wf, label=f'y_bin={y_bin}', linewidth=1.5)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Induced Current (a.u.)')
    ax.set_title(f'Varying y_bin (x_bin={x_bin}, diff_idx={diff_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: 2D grid of positions - show integral (total charge)
    ax = axes[1, 0]
    integral_map = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            integral_map[i, j] = response_template[diff_idx, i, j, :].sum()

    im = ax.imshow(integral_map, cmap='viridis', origin='lower')
    ax.set_xlabel('y_bin')
    ax.set_ylabel('x_bin')
    ax.set_title('Integrated charge vs position (diff_idx=0)')
    plt.colorbar(im, ax=ax, label='Total charge')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{integral_map[i,j]:.1f}', ha='center', va='center', color='white', fontsize=9)

    # Plot 4: Compare corner vs center
    ax = axes[1, 1]
    positions = [(0, 0), (2, 2), (4, 4), (0, 4), (4, 0)]
    for x_bin, y_bin in positions:
        wf = response_template[diff_idx, x_bin, y_bin, t_start:]
        ax.plot(t, wf, label=f'({x_bin},{y_bin})', linewidth=1.5)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Induced Current (a.u.)')
    ax.set_title('Corner vs Center positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lut_varying_position.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/lut_varying_position.png")


def plot_varying_diffusion(response_template, long_diff_template, metadata, signal_length=200):
    """Plot waveforms varying longitudinal diffusion."""

    t_tick = metadata['time_tick']
    Nt = response_template.shape[-1]

    # Fixed position at center of pixel
    x_bin, y_bin = 2, 2

    # Time axis
    t_start = Nt - signal_length
    t = np.arange(signal_length) * t_tick

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Different diffusion indices
    ax = axes[0, 0]
    diff_indices = [0, 10, 30, 50, 70, 99]
    for diff_idx in diff_indices:
        wf = response_template[diff_idx, x_bin, y_bin, t_start:]
        sigma = long_diff_template[diff_idx]
        ax.plot(t, wf, label=f'idx={diff_idx} (σ={sigma:.1f} ticks)', linewidth=1.5)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Induced Current (a.u.)')
    ax.set_title(f'Varying diffusion (x_bin={x_bin}, y_bin={y_bin})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoom on peak region
    ax = axes[0, 1]
    for diff_idx in diff_indices:
        wf = response_template[diff_idx, x_bin, y_bin, t_start:]
        sigma = long_diff_template[diff_idx]
        ax.plot(t, wf, label=f'σ={sigma:.1f}', linewidth=1.5)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Induced Current (a.u.)')
    ax.set_title('Zoom: Effect of diffusion on pulse shape')
    ax.set_xlim([10, 20])  # Zoom to peak region
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Compare low vs high diffusion (normalized)
    ax = axes[1, 0]
    for diff_idx in [0, 50, 99]:
        wf = response_template[diff_idx, x_bin, y_bin, t_start:]
        wf_norm = wf / np.max(np.abs(wf))  # Normalize to peak
        sigma = long_diff_template[diff_idx]
        ax.plot(t, wf_norm, label=f'σ={sigma:.1f} ticks', linewidth=1.5)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Normalized Current')
    ax.set_title('Normalized waveforms (peak=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative charge vs time for different diffusions
    ax = axes[1, 1]
    for diff_idx in [0, 30, 60, 99]:
        wf = response_template[diff_idx, x_bin, y_bin, t_start:]
        cumsum = np.cumsum(wf)
        sigma = long_diff_template[diff_idx]
        ax.plot(t, cumsum, label=f'σ={sigma:.1f}', linewidth=1.5)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Cumulative Charge')
    ax.set_title('Charge collection vs time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Full charge')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lut_varying_diffusion.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/lut_varying_diffusion.png")


def plot_neighbor_response(response_template, metadata, signal_length=200):
    """Plot response on neighbor pixels (demonstrates induced signal)."""

    t_tick = metadata['time_tick']
    Nt = response_template.shape[-1]

    diff_idx = 0  # No diffusion
    t_start = Nt - signal_length
    t = np.arange(signal_length) * t_tick

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Response at different distances from main pixel
    ax = axes[0]
    # Main pixel region: bins 0-4
    # First neighbor: bins 5-9
    # Second neighbor: bins 10-14
    positions = [
        (2, 2, 'Main pixel (2,2)'),
        (7, 2, '1st neighbor X (7,2)'),
        (12, 2, '2nd neighbor X (12,2)'),
        (2, 7, '1st neighbor Y (2,7)'),
        (7, 7, 'Diagonal neighbor (7,7)'),
    ]

    for x_bin, y_bin, label in positions:
        if x_bin < response_template.shape[1] and y_bin < response_template.shape[2]:
            wf = response_template[diff_idx, x_bin, y_bin, t_start:]
            ax.plot(t, wf, label=f'{label}', linewidth=1.5)

    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Induced Current (a.u.)')
    ax.set_title('Response at main vs neighbor pixel positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Integrated charge as function of distance
    ax = axes[1]
    # Sample along x-axis at y=2
    x_bins = range(0, min(25, response_template.shape[1]))
    integrals = [response_template[diff_idx, x, 2, :].sum() for x in x_bins]

    ax.bar(x_bins, integrals, alpha=0.7)
    ax.axvline(x=4.5, color='r', linestyle='--', label='Pixel boundary')
    ax.axvline(x=9.5, color='r', linestyle='--')
    ax.axvline(x=14.5, color='r', linestyle='--')
    ax.set_xlabel('x_bin')
    ax.set_ylabel('Integrated Charge')
    ax.set_title('Charge vs position (y_bin=2, diff_idx=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lut_neighbor_response.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/lut_neighbor_response.png")


def plot_full_timeline(response_template, metadata):
    """Plot the full LUT timeline to show when signal appears."""

    t_tick = metadata['time_tick']
    Nt = response_template.shape[-1]
    t = np.arange(Nt) * t_tick

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Full timeline at center position
    ax = axes[0]
    x_bin, y_bin = 2, 2
    for diff_idx in [0, 50, 99]:
        wf = response_template[diff_idx, x_bin, y_bin, :]
        sigma = diff_idx / 10  # approximate
        ax.plot(t, wf, label=f'diff_idx={diff_idx}', linewidth=1)

    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Induced Current (a.u.)')
    ax.set_title(f'Full LUT timeline (x_bin={x_bin}, y_bin={y_bin})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.axvline(x=177, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=191, color='gray', linestyle='--', alpha=0.5)
    ax.text(177, ax.get_ylim()[1]*0.9, 'Signal start', fontsize=9)
    ax.text(191, ax.get_ylim()[1]*0.9, 'Signal end', fontsize=9)

    # Plot 2: Cumulative charge over full timeline
    ax = axes[1]
    for diff_idx in [0, 50, 99]:
        wf = response_template[diff_idx, x_bin, y_bin, :]
        cumsum = np.cumsum(wf)
        ax.plot(t, cumsum, label=f'diff_idx={diff_idx}', linewidth=1.5)

    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Cumulative Charge')
    ax.set_title('Charge accumulation over full drift')
    ax.axhline(y=10, color='k', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lut_full_timeline.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/lut_full_timeline.png")


def main():
    lut_file = 'src/larndsim/detector_properties/response_44_v2a_full_tick.npz'

    print("=" * 60)
    print("LUT Waveform Visualization")
    print("=" * 60)

    # Load and build response template
    response_template, long_diff_template, metadata = load_and_build_response_template(lut_file)

    print("\nGenerating plots...")

    # Generate all plots
    plot_varying_position(response_template, metadata)
    plot_varying_diffusion(response_template, long_diff_template, metadata)
    plot_neighbor_response(response_template, metadata)
    plot_full_timeline(response_template, metadata)

    print("\nDone! All plots saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
