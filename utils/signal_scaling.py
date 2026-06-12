import cv2
import numpy as np

def get_ms_per_pixel(reference_image_path, crop_rect=(37, 119, 1876, 921), interval_ms=200.0):
    img = cv2.imread(str(reference_image_path))

    if img is None: raise RuntimeError(f"Nie udało się wczytać obrazu referencyjnego: {reference_image_path}")

    x, y, w, h = crop_rect
    img = img[y:y+h, x:x+w]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 185, 255, cv2.THRESH_BINARY)
    img_thresh[:, :35] = 0

    column_counts = np.count_nonzero(img_thresh > 0, axis=0)
    sorted_cols = np.argsort(column_counts)[::-1]

    x1 = sorted_cols[0]
    x2 = sorted_cols[1]

    distance_px = abs(x2 - x1)
    ms_per_px = interval_ms / distance_px

    return ms_per_px

def estimate_period_autocorr(counts, min_period=5, max_period=80, min_corr=0.25):
    counts = counts.astype(np.float32)
    counts -= np.mean(counts)

    if np.std(counts) > 0: counts /= np.std(counts)

    corr = np.correlate(counts, counts, mode="full")
    corr = corr[len(corr) // 2:]

    if corr[0] != 0: corr = corr / corr[0]

    search = corr[min_period:max_period + 1]
    peaks = np.where((search[1:-1] > search[:-2]) & (search[1:-1] > search[2:]))[0] + 1
    strong_peaks = peaks[search[peaks] > min_corr]

    if len(strong_peaks) > 0: lag = int(strong_peaks[0] + min_period)
    else: lag = int(np.argmax(search) + min_period)

    if 1 <= lag < len(corr) - 1:
        y0 = corr[lag - 1]
        y1 = corr[lag]
        y2 = corr[lag + 1]

        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-6: lag = lag + 0.5 * (y0 - y2) / denom

    return float(lag)


def get_ms_per_pixel_from_grid(img, paper_speed_mm_s=25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    normalized = cv2.divide(gray, background, scale=255)

    dark_enhanced = 255 - normalized

    _, grid_mask = cv2.threshold(dark_enhanced, 18, 255, cv2.THRESH_BINARY)
    column_counts = np.count_nonzero(grid_mask > 0, axis=0)
    grid_period_px = estimate_period_autocorr(column_counts, min_period=5, max_period=80)

    ms_per_small_grid = 1000.0 / paper_speed_mm_s
    ms_per_px = ms_per_small_grid / grid_period_px

    print(f"Okres kratki: {grid_period_px:.2f} px, Skala: {ms_per_px:.4f} ms/px")

    return ms_per_px

def interpolate_to_1ms(amplitude_px, ms_per_px, target_time_ms=None):
    width = len(amplitude_px)

    source_time_ms = np.arange(width) * ms_per_px
    if target_time_ms is not None:
        amplitude_1ms = np.interp(target_time_ms, source_time_ms, amplitude_px, left=0.0, right=0.0)
    else:
        target_time_ms = np.arange(0.0, np.floor(source_time_ms[-1]) + 1.0, 1.0)
        amplitude_1ms = np.interp(target_time_ms, source_time_ms, amplitude_px)
    
    # Zaokrąglenie amplitud:
    amplitude_1ms = np.rint(amplitude_1ms).astype(int)

    return target_time_ms, amplitude_1ms