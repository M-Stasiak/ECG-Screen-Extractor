from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.random_color import random_color
from utils.band_detection import detect_channel_bands_dp
from utils.signal_scaling import get_ms_per_pixel_from_grid, interpolate_to_1ms
from utils.trace_extraction import show_trace, extract_trace_dynamic_viterbi

from utils.bpm_detection import estimate_bpm_from_dataframe

SCRIPT_DIR = Path(__file__).resolve().parent

def empty_callback(value):
    pass

def process_image(signal_mask, ms_per_px, bands, name=None):
    max_width_px = max(band["roi"][1] - (band["roi"][0] + int(0.10 * (band["roi"][1] - band["roi"][0]))) for band in bands)
    max_duration_ms = (max_width_px - 1) * ms_per_px
    time_ms = np.arange(0.0, np.floor(max_duration_ms) + 1.0, 1.0)
    data = {"time_ms": time_ms}

    display_image = signal_mask.copy()
    for band in bands:
        channel_name = band["name"]
        x1, x2, y1, y2 = band["roi"]
        baseline_y = band["baseline_y"]

        channel_band = signal_mask[y1:y2, x1:x2].copy()
        baseline_local_y = baseline_y - y1

        trace_y, amplitude_px = extract_trace_dynamic_viterbi(channel_band, baseline_local_y)
        _, amplitude_1ms = interpolate_to_1ms(amplitude_px, ms_per_px, target_time_ms=time_ms)

        data[channel_name] = amplitude_1ms

        color = random_color()
        display_image = show_trace(display_image, trace_y, x_offset=x1, y_offset=y1, trace_color=color, baseline_y=baseline_local_y)
    
    if name is not None: cv2.imshow(name, display_image)

    df = pd.DataFrame(data)
    return df

def main(input_dir, output_dir, paper_speed_mm_s=25):

    image_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")

    for image_path in image_files:
        print(f"Zdjęcie: {image_path.name}")

        img_original = cv2.imread(image_path)
        img = img_original.copy()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        background = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=35, sigmaY=35)
        normalized = cv2.divide(img_gray, background, scale=255)

        _, signal_mask = cv2.threshold(255 - normalized, 55, 255, cv2.THRESH_BINARY)
        signal_mask = cv2.morphologyEx(signal_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

        height, width = signal_mask.shape

        y_start = int(0.05 * height)
        y_end = int(0.90 * height)

        x_left_start = int(0.04 * width)
        x_left_end = int(0.50 * width)

        x_right_start = int(0.53 * width)
        x_right_end = int(0.98 * width)

        left_roi = (x_left_start, x_left_end, y_start, y_end)
        right_roi = (x_right_start, x_right_end, y_start, y_end)

        left_names = ["I", "II", "III", "aVR", "aVL", "aVF"]
        right_names = ["V1", "V2", "V3", "V4", "V5", "V6"]

        left_bands = detect_channel_bands_dp(signal_mask, left_roi, n_bands=6, channel_names=left_names)
        right_bands = detect_channel_bands_dp(signal_mask, right_roi, n_bands=6, channel_names=right_names)
        bands = left_bands + right_bands

        # debug_img = cv2.cvtColor(signal_mask, cv2.COLOR_GRAY2BGR)
        # for band in bands:
        #     name = band["name"]
        #     x1, x2, y1, y2 = band["roi"]
        #     baseline_y = band["baseline_y"]

        #     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.line(debug_img, (x1, baseline_y), (x2, baseline_y), (255, 0, 0), 2)

        #     if name is not None: cv2.putText(debug_img, name, (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # plt.figure(figsize=(16, 8))
        # plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        # plt.title("Wykryte pasma kanałów")
        # plt.axis("off")
        # plt.tight_layout()
        # plt.show()

        ms_per_px = get_ms_per_pixel_from_grid(img, paper_speed_mm_s=paper_speed_mm_s)
        df = process_image(signal_mask, ms_per_px, bands, name=image_path.name)

        bpm_result = estimate_bpm_from_dataframe(df)
        if bpm_result is not None: print(f"BPM: {bpm_result['bpm']:.1f} (kanał {bpm_result['channel']}, liczba R: {len(bpm_result['r_peaks'])})")
        else: print("BPM: nie udało się wyznaczyć")

        output_path = output_dir / f"{image_path.stem}.csv"
        df.to_csv(output_path, index=False)
        print(f"Zapisano: {output_path.name}")
    
    cv2.waitKey()