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

def interpolate_to_1ms(amplitude_px, ms_per_px):
    width = len(amplitude_px)

    source_time_ms = np.arange(width) * ms_per_px
    target_time_ms = np.arange(0.0, np.floor(source_time_ms[-1]) + 1.0, 1.0)
    amplitude_1ms = np.interp(target_time_ms, source_time_ms, amplitude_px)
    
    # Zaokrąglenie amplitud:
    amplitude_1ms = np.rint(amplitude_1ms).astype(int)

    return target_time_ms, amplitude_1ms