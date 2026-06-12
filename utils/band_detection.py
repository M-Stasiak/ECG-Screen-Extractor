import numpy as np

def detect_channel_bands_dp(
    signal_mask, roi, n_bands=6, channel_names=None,
    remove_left_margin_ratio=0.12, smooth_window=21, spacing_weight=1.5, 
    position_weight=0.3, min_gap_factor=0.55, max_gap_factor=1.45
):
    height, width = signal_mask.shape
    x1, x2, y1, y2 = roi

    x1, x2 = max(0, int(x1)), min(width, int(x2))
    y1, y2 = max(0, int(y1)), min(height, int(y2))

    roi_img = signal_mask[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi_img.shape

    if roi_h <= 0 or roi_w <= 0: raise RuntimeError("Niepoprawny obszar ROI.")

    roi_img[:, :int(remove_left_margin_ratio * roi_w)] = 0
    row_score = np.count_nonzero(roi_img > 0, axis=1).astype(np.float32)

    if smooth_window % 2 == 0: smooth_window += 1

    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    row_score = np.convolve(row_score, kernel, mode="same")

    if row_score.max() > 0: row_score /= row_score.max()

    expected_spacing = roi_h / n_bands
    min_gap = max(1, int(min_gap_factor * expected_spacing))
    max_gap = max(min_gap + 1, int(max_gap_factor * expected_spacing))

    ys = np.arange(roi_h, dtype=np.float32)

    dp = np.full((n_bands, roi_h), np.inf, dtype=np.float32)
    parent = np.full((n_bands, roi_h), -1, dtype=np.int32)

    expected_y = 0.5 * expected_spacing
    dp[0] = -row_score + position_weight * ((ys - expected_y) / expected_spacing) ** 2

    for k in range(1, n_bands):
        expected_y = (k + 0.5) * expected_spacing
        position_penalty = position_weight * ((ys - expected_y) / expected_spacing) ** 2

        for y_curr in range(roi_h):
            y_prev_min = max(0, y_curr - max_gap)
            y_prev_max = y_curr - min_gap

            if y_prev_max < y_prev_min: continue

            y_prev = np.arange(y_prev_min, y_prev_max + 1)
            gaps = y_curr - y_prev

            costs = dp[k - 1, y_prev] + spacing_weight * ((gaps - expected_spacing) / expected_spacing) ** 2
            best_idx = int(np.argmin(costs))
            dp[k, y_curr] = costs[best_idx] - row_score[y_curr] + position_penalty[y_curr]
            parent[k, y_curr] = y_prev[best_idx]

    last_y = int(np.argmin(dp[-1]))

    if not np.isfinite(dp[-1, last_y]): raise RuntimeError("Nie udało się wykryć pasm kanałów.")

    centers = np.zeros(n_bands, dtype=np.int32)
    centers[-1] = last_y

    for k in range(n_bands - 1, 0, -1): centers[k - 1] = parent[k, centers[k]]
    if np.any(centers < 0): raise RuntimeError("Nie udało się odtworzyć środków pasm.")

    centers = np.sort(centers + y1)
    spacing = int(np.median(np.diff(centers)))

    if channel_names is None: channel_names = [None] * n_bands

    bands = []
    for i, center in enumerate(centers):
        y_top = max(0, center - spacing // 2) if i == 0 else (centers[i - 1] + center) // 2
        y_bottom = min(height, center + spacing // 2) if i == n_bands - 1 else (center + centers[i + 1]) // 2

        bands.append({
            "name": channel_names[i] if i < len(channel_names) else None,
            "roi": (x1, x2, int(y_top), int(y_bottom)),
            "baseline_y": int(center)
        })

    return bands