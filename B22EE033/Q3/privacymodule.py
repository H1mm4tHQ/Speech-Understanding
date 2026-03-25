import torch
import torchaudio

class PrivacyVoiceTransformer(torch.nn.Module):
    def __init__(self, pitch_shift=4, speed=1.1):
        super().__init__()
        self.pitch_shift = pitch_shift
        self.speed = speed

    def forward(self, waveform, sample_rate):
        # Pitch Shift (gender transformation)
        shifted = torchaudio.functional.pitch_shift(
            waveform, sample_rate, self.pitch_shift
        )

        # Speed perturbation (age transformation)
        new_sr = int(sample_rate * self.speed)

        stretched = torchaudio.functional.resample(
            shifted, sample_rate, new_sr
        )

        # Resample back to original SR
        output = torchaudio.functional.resample(
            stretched, new_sr, sample_rate
        )

        return output