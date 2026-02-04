from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor


@dataclass
class MelSpectrogramConfig:
    """Configuration for mel-spectrogram extraction."""
    
    sr: int = 22050  # Sample rate (RUSLAN is 44100 Hz, we resample to 22050 Hz)
    n_fft: int = 1024  # FFT size
    win_length: int = 1024  # Window length
    hop_length: int = 256  # Hop length
    n_mels: int = 80  # Number of mel bands
    f_min: float = 0.0  # Minimum frequency
    f_max: float = 8000.0  # Maximum frequency (Nyquist / 2 approximately)
    power: float = 1.0  # Power for spectrogram (1 for energy, 2 for power)
    center: bool = True  # Whether to pad signal on both sides
    pad_mode: str = "reflect"  # Padding mode
    normalized: bool = False  # Whether to normalize mel spectrogram
    norm: Optional[str] = None  # Mel filterbank normalization ("slaney" or None)
    mel_scale: str = "htk"  # Mel scale ("htk" or "slaney")


class MelSpectrogram(nn.Module):
    """
    Mel-spectrogram transform module.
    
    Extracts mel-spectrograms from raw waveforms using the same configuration
    as used during HiFi-GAN training.
    """
    
    def __init__(self, config: Optional[MelSpectrogramConfig] = None):
        super().__init__()
        
        if config is None:
            config = MelSpectrogramConfig()
        
        self.config = config
        
        # Create mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
            center=config.center,
            pad_mode=config.pad_mode,
            normalized=config.normalized,
            norm=config.norm,
            mel_scale=config.mel_scale,
        )
        
        # Mel spectrogram uses log scale for training
        self.log_eps = 1e-5
    
    def forward(self, audio: Tensor) -> Tensor:
        """
        Extract mel-spectrogram from audio waveform.
        
        Args:
            audio: Audio waveform tensor of shape (batch, time) or (time,)
            
        Returns:
            Mel-spectrogram tensor of shape (batch, n_mels, time) or (n_mels, time)
        """
        # Ensure input is at least 2D
        squeeze = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze = True
        
        # Compute mel spectrogram
        mel = self.mel_spectrogram(audio)
        
        # Apply log scaling
        mel = torch.log(torch.clamp(mel, min=self.log_eps))
        
        if squeeze:
            mel = mel.squeeze(0)
        
        return mel
    
    def inverse(self, mel: Tensor) -> Tensor:
        """
        Approximate inverse mel-spectrogram (for debugging purposes).
        Note: This is not used in HiFi-GAN training, the generator learns the inverse.
        
        Args:
            mel: Log mel-spectrogram tensor
            
        Returns:
            Approximate audio waveform (not used in practice)
        """
        raise NotImplementedError(
            "Inverse mel-spectrogram is learned by the HiFi-GAN generator, "
            "not computed analytically."
        )
    
    @property
    def hop_length(self) -> int:
        """Return hop length for calculating audio length from mel length."""
        return self.config.hop_length
    
    @property
    def sample_rate(self) -> int:
        """Return sample rate."""
        return self.config.sr
    
    @property
    def n_mels(self) -> int:
        """Return number of mel bands."""
        return self.config.n_mels


def get_mel_from_wav(audio: Tensor, mel_spec: MelSpectrogram) -> Tensor:
    """
    Utility function to extract mel-spectrogram from audio.
    
    Args:
        audio: Audio waveform tensor
        mel_spec: MelSpectrogram transform instance
        
    Returns:
        Log mel-spectrogram tensor
    """
    return mel_spec(audio)
