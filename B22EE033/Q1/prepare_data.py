from pathlib import Path
import csv

import torchaudio


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def main():
    manifest_path = DATA_DIR / "manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        src = BASE_DIR.parent.parent / row["source"]
        dst = DATA_DIR / row["audio_file"]
        if dst.exists():
            continue
        waveform, sr = torchaudio.load(str(src))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        torchaudio.save(str(dst), waveform, sr)
        print(f"Created {dst.name} from {src.name}")


if __name__ == "__main__":
    main()
