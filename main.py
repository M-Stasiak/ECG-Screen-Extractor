from pathlib import Path
import cv2
import numpy as np
import random

from utils.baseline_detection import detect_baselines
from utils.trace_extraction import extract_trace_greedy, extract_trace_dynamic, show_trace

SCRIPT_DIR = Path(__file__).resolve().parent

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def empty_callback(value):
    pass

def main():
    # cv2.namedWindow('result')
    # cv2.createTrackbar('Thresh', 'result', 60, 255, empty_callback)

    img_original = cv2.imread(SCRIPT_DIR / "data" / "P5_1.jpg")
    img = img_original.copy()

    x, y, w, h = (37, 119, 1876, 921)
    img = img[y:y+h, x:x+w]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 185, 255, cv2.THRESH_BINARY)

    # Wycięcie marginesu z lewej strony z nazwami kanałów
    img_thresh[:, :35] = 0

    baselines = detect_baselines(img_thresh)
    channel_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    display_img = img.copy()
    for channel_name, baseline_y in zip(channel_names, baselines):
        cv2.line(display_img, (0, baseline_y), (display_img.shape[1] - 1, baseline_y), (0, 255, 0), 1)
        cv2.putText(display_img, channel_name, (40, baseline_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("baselines", display_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    display_img_greedy = img.copy()
    display_img_dynamic = img.copy()
    for channel_idx, channel_name in enumerate(channel_names):
        height, width = img_thresh.shape

        baseline_y = baselines[channel_idx]
        spacing = int(np.median(np.diff(baselines)))
        search_margin = int(1.2 * spacing)

        y_top = max(0, baseline_y - search_margin)
        y_bottom = min(height, baseline_y + search_margin)

        channel_band = img_thresh[y_top:y_bottom, :]
        baseline_local_y = baseline_y - y_top

        trace_greedy, amplitude_greedy = extract_trace_greedy(channel_band, baseline_local_y)
        trace_dynamic, amplitude_dynamic = extract_trace_dynamic(channel_band, baseline_local_y)

        trace_greedy_global = trace_greedy + y_top
        trace_dynamic_global = trace_dynamic + y_top

        color = random_color()
        display_img_greedy = show_trace(display_img_greedy, trace_greedy_global, trace_color=color)
        display_img_dynamic = show_trace(display_img_dynamic, trace_dynamic_global, trace_color=color)

    cv2.imshow('greedy', display_img_greedy)
    cv2.imshow('dynamic', display_img_dynamic)
    # cv2.imwrite('greedy.png', display_img_greedy)
    # cv2.imwrite('dynamic.png', display_img_dynamic)
    cv2.waitKey()


if __name__ == "__main__":
    main()