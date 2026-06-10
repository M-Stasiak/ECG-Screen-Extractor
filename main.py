from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

def empty_callback(value):
    pass

def main():
    # cv2.namedWindow('result')
    # cv2.createTrackbar('Thresh', 'result', 60, 255, empty_callback)

    img_original = cv2.imread(SCRIPT_DIR / "data" / "P3_1.jpg")
    img = img_original.copy()

    x, y, w, h = (37, 119, 1876, 921)
    img = img[y:y+h, x:x+w]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 185, 255, cv2.THRESH_BINARY)

    # Wycięcie marginesu z lewej strony z nazwami kanałów
    img_thresh[:, :35] = 0

    mask = img_thresh > 0
    signal_per_row = np.sum(mask, axis=1).astype(np.float32)
    signal_per_row_smooth = cv2.GaussianBlur(signal_per_row.reshape(-1, 1), ksize=(1, 9), sigmaX=0).ravel()

    peak_candidates = []
    for yy in range(1, len(signal_per_row_smooth) - 1):
        if (signal_per_row_smooth[yy] >= signal_per_row_smooth[yy - 1] and signal_per_row_smooth[yy] >= signal_per_row_smooth[yy + 1]):
            peak_candidates.append(yy)


    min_distance = 45
    min_strength = 20
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

        if len(peaks) >= 16: break

    baselines = sorted(peaks)[:12]
    channel_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    for channel_name, baseline_y in zip(channel_names, baselines):
        print(channel_name, baseline_y)

    display_img = img.copy()
    for channel_name, baseline_y in zip(channel_names, baselines):
        cv2.line(display_img, (0, baseline_y), (display_img.shape[1] - 1, baseline_y), (0, 255, 0), 1)
        cv2.putText(display_img, channel_name, (40, baseline_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("baselines", display_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()