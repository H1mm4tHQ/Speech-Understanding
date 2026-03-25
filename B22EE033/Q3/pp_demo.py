import os
from pathlib import Path

import torchaudio

from privacymodule import PrivacyVoiceTransformer

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"
LOCAL_SAMPLE_MP3 = BASE_DIR / "cv-valid-test" / "sample-000001.mp3"

input_path = EXAMPLES_DIR / "before_1.wav"
output_path = EXAMPLES_DIR / "after_1.wav"

os.makedirs(EXAMPLES_DIR, exist_ok=True)

if not input_path.exists():
    if not LOCAL_SAMPLE_MP3.exists():
        raise FileNotFoundError(
            f"Missing demo input audio. Expected {input_path} or {LOCAL_SAMPLE_MP3}"
        )

    waveform, sr = torchaudio.load(str(LOCAL_SAMPLE_MP3))
    torchaudio.save(str(input_path), waveform, sr)

waveform, sr = torchaudio.load(str(input_path))

model = PrivacyVoiceTransformer(pitch_shift=5, speed=1.1)

output = model(waveform, sr)

torchaudio.save(str(output_path), output, sr)

print("Privacy transformation done.")
