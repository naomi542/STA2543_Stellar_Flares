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

This tool is designed for generating synthetic training data for time-series flare classification models (e.g. LSTMs).

"""


import numpy as np
import random
from copy import deepcopy
from altaipony.flarelc import FlareLightCurve

# ----------------------------------
# Flare Shape Templates
# ----------------------------------

def kepler_rise(t):
    """
    Generate the rise phase of a flare using a Kepler-based polynomial model.

    Parameters:
        t (np.ndarray): Time array.

    Returns:
        np.ndarray: Rise phase flux values.
    """
    return 1 + 1.941 * (t / 0.5) - 0.175 * (t / 0.5)**2 - 2.246 * (t / 0.5)**3

def kepler_decay(t):
    """
    Generate the decay phase of a flare using a Kepler-based exponential model.

    Parameters:
        t (np.ndarray): Time array.

    Returns:
        np.ndarray: Decay phase flux values.
    """
    return 0.689 * np.exp(-1.600 * t) + 0.303 * np.exp(-0.278 * t)

def kepler_flare_template(duration=1.0, time_resolution=0.00139):
    """
    Create a full flare template using Kepler-inspired rise and decay models.
    
    Parameters:
        duration (float): Total duration of the flare in days.
        time_resolution (float): Cadence between time steps.
    
    Returns:
        np.ndarray: Normalized flare shape.
    """
    t_rise = np.arange(0, duration * 0.3, time_resolution)
    t_decay = np.arange(time_resolution, duration * 0.7, time_resolution)
    flare = np.concatenate([
        kepler_rise(t_rise),
        kepler_decay(t_decay)
    ])
    return flare / np.max(flare)  # Normalize to peak at 1

def generate_scaled_kepler_flare(median_flux, time_resolution=0.00139, amp_scale=0.03):
    """
    Generate a realistic flare scaled to the median flux of the star.
    Args:
        median_flux (float): Median flux of the star.
        time_resolution (float): Time step in days. Default is ~2 minutes.
        amp_scale (float): Peak flare amplitude as fraction of median flux.
    Returns:
        flare (np.ndarray): Flare shape to inject.
        rise_len (int): Number of samples in rise.
        decay_len (int): Number of samples in decay.
    """
    # Realistic duration: 20–60 min total, 2-min cadence = 10–30 points
    total_minutes = np.random.randint(20, 60)
    total_duration_days = total_minutes / (24 * 60)
    
    flare_shape = kepler_flare_template(duration=total_duration_days, time_resolution=time_resolution)
    
    # Scale flare peak to a percentage of the median flux
    scaled_flare = flare_shape * (amp_scale * median_flux)
    
    # Split rise and decay lengths
    total_points = len(flare_shape)
    rise_len = int(total_points * 0.3)
    decay_len = total_points - rise_len
    
    return scaled_flare, rise_len, decay_len
    
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

        # Generate realistic flare
        flare, rise_len, decay_len = generate_scaled_kepler_flare(median_flux=np.nanmedian(flux))

        flare_len = len(flare)

        # Find a non-overlapping spot
        attempts = 0
        flare_idx = None
        while attempts < 100:
            idx = random.randint(0, len(synthetic_time) - flare_len - 1)
            if all((idx + i) not in used_indices for i in range(flare_len)):
                flare_idx = idx
                break
            attempts += 1

        if flare_idx is None:
            continue  # couldn't find non-overlapping spot

        for i in range(flare_len):
            synthetic_flux[flare_idx + i] += flare[i]
            binary_labels[flare_idx + i] = 1
            if i < rise_len:
                phase_labels[flare_idx + i] = 1  # Rise
            else:
                phase_labels[flare_idx + i] = 2  # Decay
            used_indices.add(flare_idx + i)
        #flare_scaled = flare * flare_amp + np.random.normal(0, 0.0001 * flare_amp, size=flare_len)

        for i in range(flare_len):
            #synthetic_flux[flare_idx + i] += flare_scaled[i]
            synthetic_flux[flare_idx + i] += flare[i] + np.random.normal(0, 0.001 * flare[i]) # already scaled
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
        
        # Merge with pre-existing real flare labels, if available
        
        # Re-run flare detection on the synthetic light curve
        
        flcd = FlareLightCurve(time=syn_time, flux=syn_flux)
        flcd.detrended_flux = flcd.flux.copy() 
        sigma = np.ones_like(flcd.flux) * 1.5
        flcd = flcd.find_flares(sigma=sigma, minsep=5)

        if flcd.flares is not None and len(flcd.flares) > 0:
            for _, flare in flcd.flares.iterrows():
                t_start, t_peak, t_stop = flare["tstart"], flare["tpeak"], flare["tstop"]
                i_start = np.searchsorted(syn_time, t_start)
                i_peak = np.searchsorted(syn_time, t_peak)
                i_stop = np.searchsorted(syn_time, t_stop)

                for i in range(i_start, i_stop):
                    if bin_lbls[i] == 0:
                        bin_lbls[i] = 1
                        if i < i_peak:
                            phase_lbls[i] = 1  # rise
                        else:
                            phase_lbls[i] = 2  # decay
        data["flare_binary_labels"] = bin_lbls
        data["flare_phase_labels"] = phase_lbls

    return synthetic_lightcurves


