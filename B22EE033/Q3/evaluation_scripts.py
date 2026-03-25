from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"

def simple_quality_score(file_path):
    audio, sr = sf.read(file_path)

    # Proxy metric: signal energy
    energy = np.mean(audio**2)

    return energy

def compute_distance(file1, file2):
    a1, _ = sf.read(file1)
    a2, _ = sf.read(file2)

    min_len = min(len(a1), len(a2))

    dist = np.mean((a1[:min_len] - a2[:min_len])**2)

    return dist

if __name__ == "__main__":
    after_file = EXAMPLES_DIR / "after_1.wav"
    before_file = EXAMPLES_DIR / "before_1.wav"

    score = simple_quality_score(after_file)
    print("DNSMOS Proxy Score:", score)

    d = compute_distance(before_file, after_file)
    print("FAD Proxy Distance:", d)
