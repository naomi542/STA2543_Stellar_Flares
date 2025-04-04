"""
inject_synthetic_flares.py

Injects synthetic, realistic-looking flares into preprocessed light curves for supervised learning.

Key Features:
--------------
* Injects flares based on Kepler-inspired flare templates (realistic rise and decay).
* Allows injection of multiple flares per light curve (configurable range).
* Prevents overlapping flare injections — tracks previously used indices to avoid re-injection at the same time steps.
* Optionally fills data gaps by:
    - Interpolating missing timestamps
    - Adding smooth stellar-like variability (Gaussian noise around the median baseline)
* Automatically labels each time step with:
    - flare_phase_labels: 0 = no flare, 1 = rise phase, 2 = decay phase
    - flare_binary_labels: 0 = no flare, 1 = flare (rise or decay)
* Injected flare amplitude scales with the median flux of the star, plus optional Gaussian noise for realism.
* Sorted output to ensure time consistency after injection and interpolation.

Typical Usage:
--------------
    from inject_flares import inject_into_lightcurves

    # processed_lightcurves is a dictionary with keys like:
    # { 'TIC_123': {'time': ..., 'detrended_flux': ...}, ... }

    synthetic_data = inject_into_lightcurves(processed_lightcurves,
                                             num_flares_range=(min_number_flares, max_number_flares),
                                             inject_in_gaps=True)

Output Format:
--------------
Each TIC entry will contain updated:
    - "time": new time array (np.ndarray)
    - "synthetic_flux": new flux array with flares added
    - "flare_binary_labels": binary mask (1 = flare)
    - "flare_phase_labels": phase labels (1 = rise, 2 = decay)

This tool is designed for generating synthetic training data for time-series flare classification models (e.g. LSTMs, CNNs, HMMs).

"""


import numpy as np
import random
from copy import deepcopy

# ----------------------------------
# Flare Shape Templates
# ----------------------------------

def kepler_rise(t):
    """Kepler-inspired rise phase of a flare."""
    return 1 + 1.941 * (t / 0.5) - 0.175 * (t / 0.5)**2 - 2.246 * (t / 0.5)**3

def kepler_decay(t):
    """Kepler-inspired decay phase of a flare."""
    return 0.689 * np.exp(-1.600 * t) + 0.303 * np.exp(-0.278 * t)

def kepler_flare_template(duration=1.0, time_resolution=0.00139):
    """
    Generate a normalized flare profile using a Kepler-based template.

    Parameters:
        duration (float): Total flare duration in days.
        time_resolution (float): Time step between flare samples (2 minute cadence = 0.00139 days).

    Returns:
        np.ndarray: A flare shape normalized to a peak of 1.
    """
    t_rise = np.arange(0, duration * 0.3, time_resolution)
    t_decay = np.arange(time_resolution, duration * 0.7, time_resolution)
    flare = np.concatenate([
        kepler_rise(t_rise),
        kepler_decay(t_decay)
    ])
    return flare / np.max(flare)  # Normalize to peak at 1

# ----------------------------------
# Utilities for Data Gaps
# ----------------------------------

def find_data_gaps(time, threshold_multiplier=5):
    """
    Identify significant time gaps in the light curve.

    Parameters:
        time (np.ndarray): Time values.
        threshold_multiplier (float): Multiplier of the median time diff to define a gap.

    Returns:
        np.ndarray: Indices where gaps start.
    """
    dt = np.diff(time)
    threshold = threshold_multiplier * np.median(dt)
    return np.where(dt > threshold)[0]

def fill_gap_with_variability(length, baseline):
    """
    Create realistic variability to fill a data gap.

    Parameters:
        length (int): Number of points to fill.
        baseline (float): Baseline flux level.

    Returns:
        np.ndarray: Synthetic noise to fill gap region.
    """
    return np.random.normal(loc=baseline, scale=0.0005 * baseline, size=length)

def interpolate_gap_time(time_start, time_end, num_points):
    """
    Linearly interpolate time values to fill a gap.

    Parameters:
        time_start (float): Time before gap.
        time_end (float): Time after gap.
        num_points (int): Number of points to interpolate.

    Returns:
        np.ndarray: Interpolated time values.
    """
    return np.linspace(time_start, time_end, num_points + 2)[1:-1]  # Exclude endpoints

# ----------------------------------
# Flare Injection into a Single Light Curve
# ----------------------------------

def inject_into_lightcurve(time, flux, num_flares=1, inject_in_gaps=True):
    """
    Inject synthetic flares into a single light curve.

    Parameters:
        time (np.ndarray): Time array.
        flux (np.ndarray): Detrended flux array.
        num_flares (int): Number of flares to inject.
        inject_in_gaps (bool): Whether to inject some flares into data gaps.

    Returns:
        tuple: synthetic_time, synthetic_flux, binary_labels, phase_labels
    """
    time = np.asarray(time)
    flux = np.asarray(flux)
    synthetic_time = time.tolist()
    synthetic_flux = flux.tolist()
    binary_labels = [0] * len(flux)
    phase_labels = [0] * len(flux)

    # Fill gaps with interpolated values + noise
    if inject_in_gaps:
        gaps = find_data_gaps(time)
        for gap_start in sorted(gaps):
            if gap_start >= len(time) - 1:
                continue
            gap_end = gap_start + 1
            dt = time[gap_end] - time[gap_start]
            expected_dt = np.median(np.diff(time))
            n_fill = int(dt / expected_dt) - 1
            if n_fill <= 0:
                continue

            new_times = interpolate_gap_time(time[gap_start], time[gap_end], n_fill)
            fill_values = fill_gap_with_variability(n_fill, baseline=np.nanmedian(flux))
            insert_idx = gap_end
            synthetic_time[insert_idx:insert_idx] = new_times.tolist()
            synthetic_flux[insert_idx:insert_idx] = fill_values.tolist()
            binary_labels[insert_idx:insert_idx] = [0] * n_fill
            phase_labels[insert_idx:insert_idx] = [0] * n_fill

    # Inject flares at non-overlapping locations so we don't mess up existing flare labels (existing flare is decaying but gets overwritten by rise of new flare)
    used_indices = set()

    for _ in range(num_flares):
        flare = kepler_flare_template(duration=0.5 + random.random() * 1.5)
        flare_len = len(flare)

        if len(synthetic_time) < flare_len + 1:
            continue  # not enough room

        attempts = 0
        max_attempts = 50
        flare_idx = None

        while attempts < max_attempts:
            idx = random.randint(0, len(synthetic_time) - flare_len - 1)
            if all((idx + i) not in used_indices for i in range(flare_len)):
                flare_idx = idx
                break
            attempts += 1

        if flare_idx is None:
            continue  # couldn't find non-overlapping spot

        flare_amp = 0.01 * np.nanmedian(flux)
        flare_scaled = flare * flare_amp + np.random.normal(0, 0.0001 * flare_amp, size=flare_len)

        for i in range(flare_len):
            synthetic_flux[flare_idx + i] += flare_scaled[i]
            binary_labels[flare_idx + i] = 1
            phase_labels[flare_idx + i] = 1 if i < flare_len // 2 else 2
            used_indices.add(flare_idx + i)  # track used index

    # Convert list to numpy array
    synthetic_time = np.array(synthetic_time)
    synthetic_flux = np.array(synthetic_flux)
    binary_labels = np.array(binary_labels)
    phase_labels = np.array(phase_labels)

    # Sort everything by time (important after filling/interpolating)
    sort_idx = np.argsort(synthetic_time)
    synthetic_time = synthetic_time[sort_idx]
    synthetic_flux = synthetic_flux[sort_idx]
    binary_labels = binary_labels[sort_idx]
    phase_labels = phase_labels[sort_idx]

    return synthetic_time, synthetic_flux, binary_labels, phase_labels
    

# ----------------------------------
# Apply Injection to Dataset
# ----------------------------------

def inject_into_lightcurves(processed_lightcurves: dict,
                             num_flares_range=(1, 3),
                             inject_in_gaps=True):
    """
    Apply flare injection across all TICs in the dataset.

    Parameters:
        processed_lightcurves (dict): Original light curve dataset.
        num_flares_range (tuple): Range of number of flares per light curve.
        inject_in_gaps (bool): Whether to sometimes inject flares in time gaps.

    Returns:
        dict: Modified dataset with injected flares and labels.
    """
    synthetic_lightcurves = deepcopy(processed_lightcurves)

    print("!!!!TEST!!!")
    for tic_id, data in synthetic_lightcurves.items():
        time = data["time"]
        flux = data["detrended_flux"]
        num_flares = random.randint(*num_flares_range)

        syn_time, syn_flux, bin_lbls, phase_lbls = inject_into_lightcurve(
            time, flux, num_flares=num_flares, inject_in_gaps=inject_in_gaps
        )

        data["time"] = syn_time
        data["synthetic_flux"] = syn_flux
        data["flare_binary_labels"] = bin_lbls
        data["flare_phase_labels"] = phase_lbls

    return synthetic_lightcurves
