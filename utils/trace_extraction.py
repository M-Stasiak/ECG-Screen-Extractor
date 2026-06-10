import numpy as np
import cv2

def extract_trace_greedy(channel_band, baseline_local_y, baseline_weight=0.15):
    height, width = channel_band.shape

    trace_y = np.full(width, np.nan, dtype=np.float32)
    previous_y = baseline_local_y
    for x in range(width):
        column = channel_band[:, x]

        ys = np.where(column > 0)[0]
        if len(ys) == 0: continue

        split_points = np.where(np.diff(ys) > 1)[0] + 1
        groups = np.split(ys, split_points)

        centers = np.array([np.mean(group) for group in groups], dtype=np.float32)
        scores = (np.abs(centers - previous_y) + baseline_weight * np.abs(centers - baseline_local_y))
        best_center = centers[np.argmin(scores)]
        trace_y[x] = best_center
        previous_y = best_center

    x_values = np.arange(width)
    valid = ~np.isnan(trace_y)
    if np.sum(valid) < 2: raise RuntimeError("Za mało punktów sygnału do interpolacji.")
    trace_y_interp = np.interp(x_values, x_values[valid], trace_y[valid])
    amplitude_px = baseline_local_y - trace_y_interp

    return trace_y_interp, amplitude_px

def extract_trace_dynamic(channel_band, baseline_local_y, max_jump=5, smooth_weight=0.15, baseline_weight=0.01, start_margin=25):
    height, width = channel_band.shape
    cost_img = 255 - channel_band
    dist = cv2.distanceTransform(cost_img, cv2.DIST_L2, 3)
    # dist_visualize = 255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imshow('Distance visualize', cv2.applyColorMap(dist_visualize, cv2.COLORMAP_INFERNO))

    dp = np.full((height, width), np.inf, dtype=np.float32)
    parent = np.full((height, width), -1, dtype=np.int32)

    ys_all = np.arange(height)
    baseline_cost = baseline_weight * np.abs(ys_all - baseline_local_y)

    start_y_top = max(0, baseline_local_y - start_margin)
    start_y_bottom = min(height, baseline_local_y + start_margin + 1)
    dp[start_y_top:start_y_bottom, 0] = (dist[start_y_top:start_y_bottom, 0] + baseline_cost[start_y_top:start_y_bottom])

    for x in range(1, width):
        for y in range(height):

            prev_y_top = max(0, y - max_jump)
            prev_y_bottom = min(height, y + max_jump + 1)
            prev_ys = np.arange(prev_y_top, prev_y_bottom)

            transition_cost = smooth_weight * np.abs(prev_ys - y)
            costs = dp[prev_y_top:prev_y_bottom, x - 1] + transition_cost
            best_idx = np.argmin(costs)
            best_previous_cost = costs[best_idx]

            dp[y, x] = (dist[y, x] + baseline_cost[y] + best_previous_cost)
            parent[y, x] = prev_y_top + best_idx
    

    trace_y = np.zeros(width, dtype=np.float32)
    y = int(np.argmin(dp[:, -1]))
    trace_y[-1] = y

    for x in range(width - 1, 0, -1):
        y = parent[y, x]
        if y < 0: y = int(baseline_local_y)
        trace_y[x - 1] = y

    amplitude_px = baseline_local_y - trace_y

    return trace_y, amplitude_px

def show_trace(img, trace_y, baseline_y=None, baseline_color=(0, 255, 0), trace_color=(0, 0, 255)):
    img = img.copy()
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    height, width = img.shape[:2]
    if baseline_y is not None:
        cv2.line(img, (0, int(round(baseline_y))), (width - 1, int(round(baseline_y))), baseline_color, 1)

    for x in range(width):
        yy = int(round(trace_y[x]))
        if 0 <= yy < height:
            cv2.circle(img, (x, yy), 1, trace_color, -1)
    
    return img