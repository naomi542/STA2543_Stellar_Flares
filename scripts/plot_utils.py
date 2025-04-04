"""
plot_utils.py

Generates and saves light curve plots for all processed TICs.
Includes tools for both raw and synthetic light curves.

Usage:
    from plot_utils import plot_raw_lightcurves, plot_synthetic_lightcurves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time


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

            # Labels and Titles
            plt.xlabel("Time - 2457000 [BKJD days]")
            plt.ylabel(r"Flux [e$^-$s$^{-1}$]")
            plt.title(f"TIC {tic_id} - Light Curve (Sector {sector})")
            plt.legend(loc=2, fontsize=13)
            plt.xlim(time_array[0], time_array[-1])
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

