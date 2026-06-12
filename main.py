from pathlib import Path
import argparse

from pipelines import screenshot_pipeline, photo_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstrakcja sygnałów EKG z zdjęć do plików CSV.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Ścieżka do folderu z obrazami wejściowymi.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Ścieżka do folderu, w którym zostaną zapisane pliki CSV.")
    parser.add_argument("--mode", choices=["screenshot", "scan"], default="screenshot", help="Tryb działania: screenshot dla zrzutów ekranów, scan dla realnych zdjęć/skanów. Domyślnie: screenshot.")
    parser.add_argument("--paper_speed", type=float, default=25.0, help="Prędkość papieru EKG w mm/s dla trybu scan.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # input_dir = SCRIPT_DIR / "input"
    # output_dir = SCRIPT_DIR / "output"

    if not input_dir.exists(): raise FileNotFoundError(f"Folder wejściowy nie istnieje: {input_dir}")
    if not input_dir.is_dir(): raise NotADirectoryError(f"Podana ścieżka wejściowa nie jest folderem: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "screenshot": screenshot_pipeline.main(input_dir, output_dir, reference_image_path=SCRIPT_DIR / "reference.jpg")
    elif args.mode == "scan": photo_pipeline.main(input_dir, output_dir, paper_speed_mm_s=args.paper_speed)