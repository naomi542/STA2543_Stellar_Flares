"""
load_data.py

Module to retrieve, detrend, and analyze TESS light curves using the Altaipony library.
This script is intended for preparing real observational data before synthetic flare injection.

Main Features:
-------------
- Automatically fetches light curves for given TIC IDs (stars_catalog.py) from MAST.
- Detrends light curves using the Savitzky-Golay filter.
- Identifies flares using Altaipony's built-in flare detection.
- Selects the sector with the highest flare activity for each TIC.
- Outputs time series data including raw flux, detrended flux, flare table, flare count, and sector metadata.

Functions:
----------
- process_tic(...): Downloads and processes light curves for one TIC, returns the cleanest/highest flare sector.
- process_star_sample(...): Iterates over a dictionary of star types and TICs to process all light curves in bulk.
- save_processed_lightcurves(...): Saves the processed dictionary of TICs and their light curves to a .pkl file.

Usage:
------
    from load_data import process_star_sample, save_processed_lightcurves

    # Provide your dictionary of star types and corresponding TICs
    processed_data = process_star_sample(star_dict)

    # Save result
    save_processed_lightcurves(processed_data)

Outputs:
--------
Each processed TIC dictionary contains:
- time: array of observation times
- flux: raw observed flux
- detrended_flux: baseline-normalized light curve
- flares: Altaipony-detected flares table
- num_flares: count of flares above 90th percentile energy
- sector: selected sector identifier (int)

Dependencies:
-------------
- numpy
- pickle
- altaipony
- astropy
- os

Notes:
------
- Only returns light curves with valid flux and at least one sector.
- Skips TICs with no valid data or retrieval issues.
- Designed to be used prior to flare injection for synthetic dataset creation.
"""


import numpy as np
import pickle
from altaipony.lcio import from_mast
from altaipony.flarelc import FlareLightCurve
import os


def process_tic(tic_id: int, cadence: str = "short", mission: str = "TESS",
                author: str = "SPOC", sector_constraint=None) -> dict:
    """
    Downloads and processes light curves for a given TIC ID.

    Parameters:
        tic_id (int): The TIC ID of the star
        cadence (str): "short" (2-min) or "long" (30-min) cadence
        mission (str): Mission name (default: "TESS")
        author (str): Light curve provider (e.g. "SPOC")
        sector_constraint (int or None): Specific sector to use (or None for all)

    Returns:
        dict containing time, flux, detrended flux, flare table, flare count, and sector.
        Returns None if no usable data was found.
    """
    try:
        # Fetch data for TIC
        flc_list = from_mast(f"TIC {tic_id}", mode="LC", c=sector_constraint,
                             cadence=cadence, mission=mission, author=author)
        
        # If no data found, skip this TIC
        if flc_list is None or len(flc_list) == 0:
            print(f" No data found... skipping...")
            return None

        # Convert single light curve to list for consistency
        if isinstance(flc_list, FlareLightCurve):
            flc_list = [flc_list]

        flare_counts = {}
        sector_data = {}

        for flc in flc_list:
            if not isinstance(flc, FlareLightCurve):
                continue
            if flc.flux is None or len(flc.flux) == 0:
                continue

            # Detrend light curve
            flcd = flc.detrend("savgol")

            # Find flares using Altaipony's detection
            flcd = flcd.find_flares()

            # Initialize binary and phase labels for each time point
            time_array = np.array(flcd.time.filled(np.nan).value, dtype=float)
            flare_binary_labels = np.zeros_like(time_array, dtype=int)
            flare_phase_labels = np.zeros_like(time_array, dtype=int)

            # Label real detected flares using Altaipony's flare table
            if flcd.flares is not None and len(flcd.flares) > 0:
                for _, flare in flcd.flares.iterrows():
                    t_start = flare["tstart"]
                    t_peak = flare["tpeak"]
                    t_stop = flare["tstop"]

                    # Get index ranges for rise and decay
                    rise_start_idx = np.searchsorted(time_array, t_start)
                    peak_idx = np.searchsorted(time_array, t_peak)
                    stop_idx = np.searchsorted(time_array, t_stop)

                    # Label rise phase as 1, decay as 2, binary mask as 1
                    flare_binary_labels[rise_start_idx:stop_idx] = 1
                    flare_phase_labels[rise_start_idx:peak_idx] = 1  # Rise
                    flare_phase_labels[peak_idx:stop_idx] = 2         # Decay


            # Count significant flares above 90th percentile energy
            if flcd.flares is not None and len(flcd.flares) > 0:
                threshold = np.percentile(flcd.flares["ed_rec"], 90)
                significant_flares = flcd.flares[flcd.flares["ed_rec"] > threshold]
                num_flares = len(significant_flares)
            else:
                num_flares = 0

            # Store sector data
            sector_data[flc.sector] = {
                
                "flare_binary_labels": flare_binary_labels,
                "flare_phase_labels": flare_phase_labels,
                "time": np.array(flcd.time.filled(np.nan).value, dtype=float),
                "flux": np.array(flcd.flux.filled(np.nan), dtype=float),
                "detrended_flux": np.array(flcd.detrended_flux.filled(np.nan), dtype=float),
                "flares": flcd.flares.copy() if flcd.flares is not None else None,
                "num_flares": num_flares
            }
            flare_counts[flc.sector] = num_flares

        # Pick sector with most flares (or longest if tied)
        if flare_counts:
            best_sector = max(flare_counts, key=lambda s: (flare_counts[s], len(sector_data[s]["time"])))
            result = sector_data[best_sector]
            result["sector"] = best_sector
            return result
        else:
            return None

    except Exception as e:
        print(f"Failed to process TIC {tic_id}: {e}")
        return None


def process_star_sample(stars_dict: dict, cadence="short", mission="TESS",
                        author="SPOC", sector_constraint=None) -> dict:
    """
    Processes a full sample of stars grouped by spectral type.

    Parameters:
        stars_dict (dict): Dictionary with keys as star types and values as TIC ID lists
        cadence, mission, author, sector_constraint: Same as `process_tic`

    Returns:
        Dictionary of TIC IDs mapped to their processed light curve data.
    """
    processed = {}

    for star_type, tic_ids in stars_dict.items():
        print(f"\nProcessing {star_type} stars...")
        for tic_id in tic_ids:
            print(f"TIC {tic_id}")
            result = process_tic(tic_id, cadence, mission, author, sector_constraint)
            if result is not None:
                processed[tic_id] = result
                print(f"Stored Sector {result['sector']} with {result['num_flares']} flares.")
            else:
                print(f"No valid data for TIC {tic_id}")

    return processed


def save_processed_lightcurves(data: dict, filepath: str = None):
    """
    Saves processed light curve data to a pickle file.

    Parameters:
        data (dict): Dictionary of TIC IDs to light curve data
        filepath (str): Path to save .pkl file
    """
    if filepath is None:
        # Build relative path from current file to root/data/
        root_dir = os.path.dirname(os.path.dirname(__file__))  # go up from /scripts/
        filepath = os.path.join(root_dir, "data", "processed_lightcurves.pkl")

    # Ensure the folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved light curves to {filepath}")

