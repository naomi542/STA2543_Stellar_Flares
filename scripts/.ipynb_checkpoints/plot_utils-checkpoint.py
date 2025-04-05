"""
plot_utils.py

Utilities for plotting raw and synthetic light curves for TESS stars.

Functions:
- plot_raw_lightcurves(...): Save plots of raw and detrended flux.
- plot_synthetic_lightcurves(...): Save plots of synthetic flux with injected flares.
- inspect_synthetic_lightcurve(...): Visualize flare phases and smoothed baselines for a specific TIC.

Usage:
    from plot_utils import plot_raw_lightcurves, plot_synthetic_lightcurves, inspect_synthetic_lightcurve
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.signal import savgol_filter


def plot_raw_lightcurves(processed_lightcurves: dict, output_dir: str = "../data/figures/raw"):
    """
    Loops through all TICs in the processed light curve dictionary,
    plots raw + detrended flux, and saves figures as PNGs.

    Parameters:
        processed_lightcurves (dict): Output from process_star_sample()
        output_dir (str): Folder to save light curve plots
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure folder exists
    print("Inside plot_raw_lightcurves function")


    for tic_id, processed_data in processed_lightcurves.items():
        try:
            print(f"TIC {tic_id}: Preparing plot...")

            # Extract Time Array
            time_data = processed_data["time"]
            if isinstance(time_data, np.ndarray) and isinstance(time_data[0], Time):
                time_array = np.array([t.value for t in time_data])
            else:
                time_array = np.asarray(time_data, dtype=float)

            # Extract Flux Arrays
            flux_array = np.asarray(processed_data["flux"], dtype=float)
            detrended_flux_array = np.asarray(processed_data["detrended_flux"], dtype=float)
            df_flares = processed_data.get("flares", None)
            sector = processed_data.get("sector", "Unknown")

            num_flares = len(df_flares) if df_flares is not None else 0
            print(f"TIC {tic_id} (Sector {sector}): {num_flares} flares detected.")

            # Plot Setup
            plt.figure(figsize=(12, 5))
            plt.plot(time_array, flux_array / np.nanmedian(flux_array) + 0.1, "r", label="PDCSAP_FLUX")
            plt.plot(time_array, detrended_flux_array / np.nanmedian(detrended_flux_array), "b", label="Detrended Flux")

            # Dynamically set y-limits based on flux range
            ymin = np.nanmin([flux_array.min(), detrended_flux_array.min()]) / np.nanmedian(detrended_flux_array)
            ymax = np.nanmax([flux_array.max(), detrended_flux_array.max()]) / np.nanmedian(detrended_flux_array)


            # Labels and Titles
            plt.xlabel("Time - 2457000 [BKJD days]")
            plt.ylabel(r"Flux [e$^-$s$^{-1}$]")
            plt.title(f"TIC {tic_id} - Light Curve (Sector {sector})")
            plt.legend(loc=2, fontsize=13)
            plt.xlim(time_array[0], time_array[-1])
            # Pad by 5% for nice margins
            yrange = ymax - ymin
            if yrange < 0.05:
                plt.ylim(0.95, 1.05)
            plt.ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)
            plt.ylim(0.95, 1.30)

            # Save Figure
            save_path = os.path.join(output_dir, f"TIC_{tic_id}_Sector_{sector}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved plot to {save_path}")

        except Exception as e:
            print(f"Failed to plot TIC {tic_id}: {e}")


def plot_synthetic_lightcurves(synthetic_lightcurves: dict, output_dir: str = "../data/figures/synthetic"):
    """
    Plot synthetic light curves with injected flares.
    Overlays binary flare regions and phase labels.

    Parameters:
        synthetic_lightcurves (dict): Dictionary with injected flare info
        output_dir (str): Folder to save synthetic light curve plots
    """
    os.makedirs(output_dir, exist_ok=True)

    for tic_id, data in synthetic_lightcurves.items():
        try:
            time = np.asarray(data["time"], dtype=float)
            flux = np.asarray(data["synthetic_flux"], dtype=float)
            flare_mask = np.asarray(data["flare_binary_labels"], dtype=int)
            phase_mask = np.asarray(data["flare_phase_labels"], dtype=int)
            sector = data.get("sector", "Unknown")

            plt.figure(figsize=(12, 5))
            plt.plot(time, flux, color="black", linewidth=1, label="Synthetic Flux")

            # Overlay rising phase
            rise_mask = (phase_mask == 1)
            plt.scatter(time[rise_mask], flux[rise_mask], color="orange", s=6, label="Flare Rise")

            # Overlay decay phase
            decay_mask = (phase_mask == 2)
            plt.scatter(time[decay_mask], flux[decay_mask], color="blue", s=6, label="Flare Decay")

            # Labels and styling
            plt.xlabel("Time - 2457000 [BKJD days]")
            plt.ylabel("Synthetic Flux")
            plt.title(f"TIC {tic_id} - Synthetic Flare Injection (Sector {sector})")
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"TIC_{tic_id}_synthetic.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved synthetic flare plot: {save_path}")

        except Exception as e:
            print(f"Failed to plot synthetic light curve for TIC {tic_id}: {e}")


def inspect_synthetic_lightcurve(tic_id, synthetic_lightcurves, show_baseline=True, show_overlay=True):
    """
    Plots a synthetic light curve for a given TIC ID.
    
    Parameters:
        tic_id (str or int): The TIC ID to visualize.
        synthetic_lightcurves (dict): Dictionary of light curves loaded from your synthetic dataset.
        show_baseline (bool): Whether to plot the smoothed baseline alongside the original flux.
        show_overlay (bool): Whether to overlay flare phase labels (1 = rise, 2 = decay).
    """
    example = synthetic_lightcurves[tic_id]
    time = example["time"]
    flux = example["synthetic_flux"]
    phase_labels = example["flare_phase_labels"]

    # Plot original + baseline
    if show_baseline:
        baseline = savgol_filter(flux, window_length=101, polyorder=3)

        plt.figure(figsize=(12, 4))
        plt.plot(time, flux, color='black', linewidth=0.7, label='Synthetic Flux')
        plt.plot(time, baseline, color='red', linewidth=1.2, label='Smoothed Baseline')
        plt.title(f"TIC {tic_id} — Original vs Smoothed")
        plt.xlabel("Time [days]")
        plt.ylabel("Flux")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Overlay phase labels
    if show_overlay:
        rise_times = time[np.array(phase_labels) == 1]
        rise_flux = flux[np.array(phase_labels) == 1]

        decay_times = time[np.array(phase_labels) == 2]
        decay_flux = flux[np.array(phase_labels) == 2]

        plt.figure(figsize=(12, 4))
        plt.plot(time, flux, 'k', label="Flux")
        plt.plot(rise_times, rise_flux, 'orange', linestyle='None', marker='o', markersize=3, label="Rise")
        plt.plot(decay_times, decay_flux, 'blue', linestyle='None', marker='o', markersize=3, label="Decay")
        plt.legend()
        plt.title(f"TIC {tic_id} — Flare Phase Overlay")
        plt.xlabel("Time [days]")
        plt.ylabel("Flux")
        plt.tight_layout()
        plt.show()