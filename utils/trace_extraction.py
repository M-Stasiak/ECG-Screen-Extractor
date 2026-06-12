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


def extract_trace_dynamic_viterbi(
    channel_band, baseline_local_y,
    baseline_weight=0.004, endpoint_weight=0.04, interval_gap_weight=1.0, slope_weight=0.08,
    missing_column_weight=0.8, vertical_peak_height=5,
):
    height, width = channel_band.shape

    mask = (channel_band > 0).astype(np.uint8) * 255
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8), iterations=1)

    candidates_per_x = []
    valid_xs = []

    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) == 0:
            candidates_per_x.append(None)
            continue

        split_points = np.where(np.diff(ys) > 1)[0] + 1
        groups = np.split(ys, split_points)

        candidates = []
        for group in groups:
            y_top = float(group[0])
            y_bottom = float(group[-1])
            y_center = float(np.mean(group))
            h = y_bottom - y_top + 1

            if h >= vertical_peak_height:
                if abs(y_top - baseline_local_y) > abs(y_bottom - baseline_local_y): y_trace = y_top
                else: y_trace = y_bottom
            else:
                y_trace = y_center

            candidates.append({
                "top": y_top,
                "bottom": y_bottom,
                "center": y_center,
                "trace": y_trace
            })

        candidates_per_x.append(candidates)
        valid_xs.append(x)

    valid_xs = np.array(valid_xs, dtype=np.int32)
    if len(valid_xs) < 2: raise RuntimeError("Za mało punktów sygnału do ekstrakcji.")


    first_x = valid_xs[0]
    first_candidates = candidates_per_x[first_x]

    dp_prev = np.zeros(len(first_candidates), dtype=np.float32)
    for i, cand in enumerate(first_candidates):
        dp_prev[i] = endpoint_weight * abs(cand["trace"] - baseline_local_y) + baseline_weight * abs(cand["center"] - baseline_local_y)


    parents = []
    for idx in range(1, len(valid_xs)):
        x_prev = valid_xs[idx - 1]
        x_curr = valid_xs[idx]

        prev_candidates = candidates_per_x[x_prev]
        curr_candidates = candidates_per_x[x_curr]

        dx = max(1, x_curr - x_prev)

        dp_curr = np.full(len(curr_candidates), np.inf, dtype=np.float32)
        parent_curr = np.full(len(curr_candidates), -1, dtype=np.int32)

        for j, curr in enumerate(curr_candidates):
            curr_top = curr["top"]
            curr_bottom = curr["bottom"]
            curr_center = curr["center"]
            curr_trace = curr["trace"]

            costs = np.zeros(len(prev_candidates), dtype=np.float32)
            for i, prev in enumerate(prev_candidates):
                prev_top = prev["top"]
                prev_bottom = prev["bottom"]
                prev_center = prev["center"]
                prev_trace = prev["trace"]

                if prev_bottom < curr_top: interval_gap = max(0, curr_top - prev_bottom - 1)
                elif curr_bottom < prev_top: interval_gap = max(0, prev_top - curr_bottom - 1)
                else: interval_gap = 0.0

                slope = abs(curr_trace - prev_trace) / dx
                transition_cost = interval_gap_weight * interval_gap + slope_weight * slope + missing_column_weight * max(0, dx - 1)
                baseline_cost = baseline_weight * abs(curr_center - baseline_local_y)
                
                costs[i] = dp_prev[i] + transition_cost + baseline_cost

            best_prev_idx = int(np.argmin(costs))

            dp_curr[j] = costs[best_prev_idx]
            parent_curr[j] = best_prev_idx

        dp_prev = dp_curr
        parents.append(parent_curr)

    last_x = valid_xs[-1]
    last_candidates = candidates_per_x[last_x]

    # Wybór odpowiedniego końca
    best_score = np.inf
    best_path_indices = None

    allowed_gap = 2.0
    gap_weight = 20.0
    endpoint_weight_final = endpoint_weight * 0.25

    for end_idx, end_cand in enumerate(last_candidates):
        if not np.isfinite(dp_prev[end_idx]): continue

        path_indices_test = np.zeros(len(valid_xs), dtype=np.int32)
        path_indices_test[-1] = end_idx

        valid_path = True
        for idx in range(len(valid_xs) - 1, 0, -1):
            prev_idx = parents[idx - 1][path_indices_test[idx]]

            if prev_idx < 0:
                valid_path = False
                break

            path_indices_test[idx - 1] = prev_idx

        if not valid_path: continue

        path_penalty = 0.0
        for idx in range(1, len(valid_xs)):
            x_prev = valid_xs[idx - 1]
            x_curr = valid_xs[idx]

            prev = candidates_per_x[x_prev][path_indices_test[idx - 1]]
            curr = candidates_per_x[x_curr][path_indices_test[idx]]

            if prev["bottom"] < curr["top"]: gap = curr["top"] - prev["bottom"]
            elif curr["bottom"] < prev["top"]: gap = prev["top"] - curr["bottom"]
            else: gap = 0.0

            gap_excess = max(0.0, gap - allowed_gap)
            path_penalty += gap_weight * gap_excess * gap_excess

        end_penalty = endpoint_weight_final * abs(end_cand["trace"] - baseline_local_y)
        score = dp_prev[end_idx] + path_penalty + end_penalty

        if score < best_score:
            best_score = score
            best_path_indices = path_indices_test.copy()

    if best_path_indices is None: raise RuntimeError("Nie udało się wybrać poprawnego końca ścieżki.")

    path_indices = best_path_indices

    sparse_y = np.zeros(len(valid_xs), dtype=np.float32)
    for idx, x in enumerate(valid_xs):
        cand = candidates_per_x[x][path_indices[idx]]
        sparse_y[idx] = cand["trace"]


    x_values = np.arange(width)
    trace_y = np.interp(x_values, valid_xs, sparse_y)

    amplitude_px = baseline_local_y - trace_y

    return trace_y, amplitude_px


def show_trace_2(img, trace_y, baseline_y=None, baseline_color=(0, 255, 0), trace_color=(0, 0, 255)):
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

def show_trace(img, trace_y, x_offset=0, y_offset=0, baseline_y=None, baseline_color=(0, 255, 0), trace_color=(0, 0, 255)):
    img = img.copy()
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    height, width = img.shape[:2]
    if baseline_y is not None:
        baseline_global_y = int(round(y_offset + baseline_y))
        cv2.line(img, (x_offset, baseline_global_y), (min(width - 1, x_offset + len(trace_y) - 1), baseline_global_y), baseline_color, 1)

    for x_local in range(len(trace_y)):
        x_global = x_offset + x_local
        y_global = y_offset + int(round(trace_y[x_local]))

        if 0 <= x_global < width and 0 <= y_global < height:
            cv2.circle(img, (x_global, y_global), 1, trace_color, -1)

    return img