from pathlib import Path

import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.random_color import random_color
from utils.baseline_detection import detect_baselines
from utils.signal_scaling import get_ms_per_pixel, interpolate_to_1ms
from utils.trace_extraction import show_trace, extract_trace_greedy, extract_trace_dynamic, extract_trace_dynamic_viterbi

from utils.bpm_detection import estimate_bpm_from_dataframe

SCRIPT_DIR = Path(__file__).resolve().parent


def empty_callback(value):
    pass

def process_image(img, ms_per_px, name=None):
    baselines = detect_baselines(img)
    channel_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    data = {}
    display_img_viterbi = img.copy()
    for channel_idx, channel_name in enumerate(channel_names):
        height, width = img.shape
        baseline_y = baselines[channel_idx]

        spacing = int(np.median(np.diff(baselines)))
        search_margin = int(1.5 * spacing)

        y_top = max(0, baseline_y - search_margin)
        y_bottom = min(height, baseline_y + search_margin)

        channel_band = img[y_top:y_bottom, :]
        baseline_local_y = baseline_y - y_top

        trace_viterbi, amplitude_viterbi = extract_trace_dynamic_viterbi(channel_band, baseline_local_y)
        time_ms, amplitude_1ms = interpolate_to_1ms(amplitude_viterbi, ms_per_px)

        if "time_ms" not in data: data["time_ms"] = time_ms
        data[channel_name] = amplitude_1ms
        trace_viterbi_global = trace_viterbi + y_top
        # trace_viterbi_global = np.pad(trace_viterbi_global, (display_img_viterbi.shape[1] - len(trace_viterbi_global), 0), constant_values=-1)
        
        color = random_color()
        display_img_viterbi = show_trace(display_img_viterbi, trace_viterbi_global, trace_color=color, baseline_y=baseline_y)
    
    if name is not None: cv2.imshow(name, display_img_viterbi)

    df = pd.DataFrame(data)
    return df

def main(input_dir, output_dir, reference_image_path):

    ms_per_px = get_ms_per_pixel(reference_image_path)

    image_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")

    for image_path in image_files:
        print(f"Zdjęcie: {image_path.name}")

        img_original = cv2.imread(image_path)
        img = img_original.copy()

        x, y, w, h = (37, 119, 1876, 921)
        img = img[y:y+h, x:x+w]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 185, 255, cv2.THRESH_BINARY)

        # Wycięcie marginesu z lewej strony z nazwami kanałów
        img_thresh[:, :35] = 0
        # img_thresh = img_thresh[:, 35:]

        df = process_image(img_thresh, ms_per_px, name=image_path.name)

        bpm_result = estimate_bpm_from_dataframe(df)
        if bpm_result is not None: print(f"BPM: {bpm_result['bpm']:.1f} (kanał {bpm_result['channel']}, liczba R: {len(bpm_result['r_peaks'])})")
        else: print("BPM: nie udało się wyznaczyć")

        output_path = output_dir / f"{image_path.stem}.csv"
        df.to_csv(output_path, index=False)
        print(f"Zapisano: {output_path.name}")
    
    cv2.waitKey()