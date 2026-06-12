from pathlib import Path

import argparse
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

def smooth_signal(signal, window_size=9):
    return signal.rolling(window=window_size, center=True, min_periods=1).mean()

def main(csv_dir):
    csv_files = sorted(p for p in csv_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    channel_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    for csv_path in csv_files:
        print(f"Plik: {csv_path.name}")

        df = pd.read_csv(csv_path)
        time_ms = df["time_ms"]

        fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(16, 18), sharex=True)
        fig.suptitle(csv_path.name, fontsize=16)

        for ax, channel_name in zip(axes, channel_names):
            if channel_name not in df.columns:
                ax.set_visible(False)
                continue

            ax.plot(time_ms, df[channel_name])
            ax.set_ylabel(channel_name)
            ax.grid(True)

        axes[-1].set_xlabel("Czas [ms]")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

        # fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(16, 18), sharex=True)
        # fig.suptitle(csv_path.name + " - Wygładzony", fontsize=16)

        # for ax, channel_name in zip(axes, channel_names):
        #     if channel_name not in df.columns:
        #         ax.set_visible(False)
        #         continue

        #     smoothed_signal = df[channel_name].rolling(window=7, center=True, min_periods=1).mean()

        #     ax.plot(time_ms, smoothed_signal)
        #     ax.set_ylabel(channel_name)
        #     ax.grid(True)

        # axes[-1].set_xlabel("Czas [ms]")

        # plt.tight_layout(rect=[0, 0, 1, 0.97])
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wizualizacja zapisanych CSV")
    parser.add_argument("--input_dir", required=True, type=Path, help="Ścieżka do folderu z plikami CSV.")
    args = parser.parse_args()

    input_dir = args.input_dir
    # input_dir = SCRIPT_DIR / "output"

    main(input_dir)