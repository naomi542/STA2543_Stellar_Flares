"""
summarize.py

Functions to summarize flare detection stats from processed light curves.
"""

def summarize_flare_detection(processed_lightcurves: dict, stars: dict):
    """
    Prints a summary of flare detections across TICs and star types.

    Parameters:
        processed_lightcurves (dict): TIC ID -> processed light curve data
        stars (dict): Star type -> list of TIC IDs
    """
    print("\n Summary of Processed Stars")
    total_stars = len(processed_lightcurves)
    print(f"Total stars processed: {total_stars}")

    # Count flares for each TIC
    flare_counts = {
        tic: len(data["flares"]) if data["flares"] is not None else 0
        for tic, data in processed_lightcurves.items()
    }

    # Total stars with at least one flare
    stars_with_flares = sum(1 for count in flare_counts.values() if count > 0)
    print(f"Stars with at least one detected flare: {stars_with_flares}")

    # Top 5 flare producers
    sorted_flares = sorted(flare_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n Top Stars with Most Flares:")
    for tic, count in sorted_flares[:5]:
        print(f"  TIC {tic}: {count} flares")

    # Count of flare-producing stars by star type
    stars_with_flares_count = {}
    for star_type, tic_ids in stars.items():
        count_with_flares = sum(
            1 for tic_id in tic_ids
            if tic_id in processed_lightcurves
            and processed_lightcurves[tic_id]["flares"] is not None
            and len(processed_lightcurves[tic_id]["flares"]) > 0
        )
        stars_with_flares_count[star_type] = count_with_flares

    print("\n Flare Detection Summary by Star Type")
    for star_type, count in stars_with_flares_count.items():
        total = len(stars[star_type])
        print(f"  {star_type}: {count} out of {total} stars had at least one flare")

    # Additional total flare/no flare stats
    total_with_flares = sum(1 for v in flare_counts.values() if v > 0)
    total_without_flares = total_stars - total_with_flares

    print("\n Final Flare Stats:")
    print(f"  Stars with Flares:    {total_with_flares}")
    print(f"  Stars without Flares: {total_without_flares}")
