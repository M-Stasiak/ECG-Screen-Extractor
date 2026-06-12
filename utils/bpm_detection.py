import numpy as np


def moving_average(signal, window):
    window = int(window)

    if window < 2: return signal.copy()
    if window > len(signal): window = len(signal)

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal, kernel, mode="same")


def detect_r_peaks(time_ms, signal, min_distance_ms=300, threshold_ratio=0.45):
    signal = np.asarray(signal, dtype=np.float32)
    time_ms = np.asarray(time_ms, dtype=np.float32)

    if len(signal) < 3: return np.array([], dtype=int)

    signal = signal - moving_average(signal, 200)
    signal = moving_average(signal, 9)

    if abs(np.min(signal)) > abs(np.max(signal)): signal = -signal

    signal_min = np.median(signal)
    signal_max = np.max(signal)

    if signal_max <= signal_min: return np.array([], dtype=int)

    threshold = signal_min + threshold_ratio * (signal_max - signal_min)
    candidates = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:]) & (signal[1:-1] > threshold))[0] + 1

    if len(candidates) == 0: return np.array([], dtype=int)

    candidates = candidates[np.argsort(signal[candidates])[::-1]]

    selected = []
    for candidate in candidates:
        candidate_time = time_ms[candidate]
        if all(abs(candidate_time - time_ms[peak]) >= min_distance_ms for peak in selected): selected.append(candidate)

    selected = np.array(sorted(selected), dtype=int)

    return selected


def calculate_bpm_from_peaks(time_ms, r_peaks):
    if len(r_peaks) < 2: return None

    rr_intervals_ms = np.diff(time_ms[r_peaks])
    rr_intervals_ms = rr_intervals_ms[(rr_intervals_ms >= 300) & (rr_intervals_ms <= 2000)]

    if len(rr_intervals_ms) == 0: return None

    bpm = 60000.0 / np.mean(rr_intervals_ms)

    return bpm


def estimate_bpm_from_dataframe(df, preferred_channels=None):
    if preferred_channels is None:
        preferred_channels = ["II", "I", "V5", "V4", "V3", "V2", "V1", "III", "aVF", "aVL", "aVR"]

    time_ms = df["time_ms"].to_numpy(dtype=np.float32)

    results = []
    for channel_name in preferred_channels:
        if channel_name not in df.columns: continue

        signal = df[channel_name].to_numpy(dtype=np.float32)

        r_peaks = detect_r_peaks(time_ms, signal)
        bpm = calculate_bpm_from_peaks(time_ms, r_peaks)

        if bpm is None: continue

        results.append({
            "channel": channel_name,
            "bpm": float(bpm),
            "r_peaks": r_peaks,
            "r_count": len(r_peaks),
            "rr_count": len(r_peaks) - 1
        })
    
    if len(results) == 0: return None

    bpm_values = np.array([r["bpm"] for r in results], dtype=np.float32)
    median_bpm = np.median(bpm_values)

    filtered_results = [r for r in results if abs(r["bpm"] - median_bpm) <= 0.20 * median_bpm]
    if len(filtered_results) == 0: filtered_results = results

    weights = np.array([max(1, r["rr_count"]) for r in filtered_results], dtype=np.float32)
    filtered_bpm = np.array([r["bpm"] for r in filtered_results], dtype=np.float32)
    final_bpm = np.average(filtered_bpm, weights=weights)

    # print("BPM z kanałów:")
    # used_channel_names = {r["channel"] for r in filtered_results}
    # for r in results:
    #     status = "użyty" if r["channel"] in used_channel_names else "odrzucony"
    #     print(f"{r['channel']}: {r['bpm']:.1f} BPM, R={r['r_count']}, {status}")

    return final_bpm