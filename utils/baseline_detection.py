import cv2
import numpy as np

def detect_baselines(img, expected_count=12, max_peaks=16, min_distance=45, min_strength=20, blur_kernel_size=9):
    mask = img > 0
    signal_per_row = np.sum(mask, axis=1).astype(np.float32)
    signal_per_row_smooth = cv2.GaussianBlur(signal_per_row.reshape(-1, 1), ksize=(1, blur_kernel_size), sigmaX=0).ravel()

    peak_candidates = []
    for yy in range(1, len(signal_per_row_smooth) - 1):
        if (signal_per_row_smooth[yy] >= signal_per_row_smooth[yy - 1] and signal_per_row_smooth[yy] >= signal_per_row_smooth[yy + 1]):
            peak_candidates.append(yy)


    peaks = []
    for yy in sorted(peak_candidates, key=lambda yy: signal_per_row_smooth[yy], reverse=True):
        if signal_per_row_smooth[yy] < min_strength: continue

        too_close = False
        for existing_peak in peaks:
            if abs(yy - existing_peak) < min_distance:
                too_close = True
                break

        if too_close: continue

        peaks.append(yy)

        if len(peaks) >= max_peaks: break

    baselines = sorted(peaks)[:expected_count]

    if len(baselines) < expected_count:
        raise RuntimeError(f"Wykryto za mało linii izoelektrycznych: {len(baselines)} zamiast {expected_count}")
    
    return baselines