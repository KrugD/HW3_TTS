from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.datasets.collate import InferenceCollator


class CustomDirDataset(Dataset):
    """
    Custom directory dataset for HiFi-GAN inference.
    
    Args:
        audio_dir: Path to directory containing audio files
        mel_spec_transform: MelSpectrogram transform instance
        target_sr: Target sample rate (default: 22050)
        resynthesize: If True, extract mel from ground-truth audio (default: True)
        mel_dir: Path to pre-computed mel-spectrograms (if resynthesize=False)
        limit: Limit number of samples
        audio_extension: Audio file extension to look for
    """
    
    def __init__(
        self,
        audio_dir: str,
        mel_spec_transform: Optional[MelSpectrogram] = None,
        target_sr: int = 22050,
        resynthesize: bool = True,
        mel_dir: Optional[str] = None,
        limit: Optional[int] = None,
        audio_extension: str = "wav",
    ):
        super().__init__()
        
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr
        self.resynthesize = resynthesize
        self.audio_extension = audio_extension
        
        # Handle directory structure - check both direct and audio/ subdirectory
        if (self.audio_dir / "audio").exists():
            self.audio_dir = self.audio_dir / "audio"
        
        # Create mel spectrogram transform if not provided
        # Always keep on CPU for multiprocessing
        if mel_spec_transform is None:
            config = MelSpectrogramConfig(sr=target_sr)
            mel_spec_transform = MelSpectrogram(config)
        # Ensure transform is on CPU for multiprocessing
        self.mel_spec_transform = mel_spec_transform.cpu()
        
        # Find audio files
        self.audio_paths = self._find_audio_files()
        
        if limit is not None:
            self.audio_paths = self.audio_paths[:limit]
        
        # Handle mel directory for pre-computed mels
        self.mel_dir = Path(mel_dir) if mel_dir is not None else None
        
        print(f"CustomDirDataset: {len(self.audio_paths)} samples, resynthesize={resynthesize}")
        
        # Collate function for inference
        self.collate_fn = InferenceCollator()
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the directory."""
        # Filter out macOS resource fork files (starting with ._)
        audio_files = sorted([
            f for f in self.audio_dir.glob(f"*.{self.audio_extension}")
            if not f.name.startswith("._")
        ])
        
        if len(audio_files) == 0:
            # Try without extension filter
            audio_files = sorted([
                f for f in self.audio_dir.iterdir() 
                if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']
                and not f.name.startswith("._")
            ])
        
        if len(audio_files) == 0:
            raise FileNotFoundError(
                f"No audio files found in {self.audio_dir}. "
                f"Looking for files with extension: {self.audio_extension}"
            )
        
        return audio_files
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - audio: Ground-truth audio waveform (for comparison)
                - mel: Mel-spectrogram for synthesis
                - audio_path: Original audio file path
                - file_name: Base file name without extension
        """
        audio_path = self.audio_paths[idx]
        file_name = audio_path.stem
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)
        
        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        # Normalize audio
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()
        
        if self.resynthesize:
            # Extract mel from ground-truth audio
            mel = self.mel_spec_transform(audio)
        else:
            # Load pre-computed mel-spectrogram
            if self.mel_dir is not None:
                mel_path = self.mel_dir / f"{file_name}.npy"
                if mel_path.exists():
                    mel = torch.from_numpy(np.load(mel_path)).float()
                else:
                    mel_path = self.mel_dir / f"{file_name}.pt"
                    if mel_path.exists():
                        mel = torch.load(mel_path)
                    else:
                        # Fallback to extracting from audio
                        print(f"Warning: Pre-computed mel not found for {file_name}, extracting from audio")
                        mel = self.mel_spec_transform(audio)
            else:
                # No mel directory provided, extract from audio
                mel = self.mel_spec_transform(audio)
        
        return {
            "audio": audio,
            "mel": mel,
            "audio_path": str(audio_path),
            "file_name": file_name,
        }


class MelDataset(Dataset):
    """
    Dataset for loading pre-computed mel-spectrograms directly.
    
    Useful for inference when mel-spectrograms are already computed
    (e.g., from an acoustic model like Tacotron2).
    
    Args:
        mel_dir: Path to directory containing mel-spectrogram files (.npy or .pt)
        limit: Limit number of samples
    """
    
    def __init__(
        self,
        mel_dir: str,
        limit: Optional[int] = None,
    ):
        super().__init__()
        
        self.mel_dir = Path(mel_dir)
        
        # Find mel files
        self.mel_paths = self._find_mel_files()
        
        if limit is not None:
            self.mel_paths = self.mel_paths[:limit]
        
        print(f"MelDataset: {len(self.mel_paths)} samples")
        
        self.collate_fn = InferenceCollator()
    
    def _find_mel_files(self) -> List[Path]:
        """Find all mel-spectrogram files."""
        mel_files = []
        for ext in [".npy", ".pt"]:
            mel_files.extend(self.mel_dir.glob(f"*{ext}"))
        
        mel_files = sorted(mel_files)
        
        if len(mel_files) == 0:
            raise FileNotFoundError(
                f"No mel-spectrogram files (.npy or .pt) found in {self.mel_dir}"
            )
        
        return mel_files
    
    def __len__(self) -> int:
        return len(self.mel_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single mel-spectrogram sample."""
        mel_path = self.mel_paths[idx]
        file_name = mel_path.stem
        
        # Load mel-spectrogram
        if mel_path.suffix == ".npy":
            mel = torch.from_numpy(np.load(mel_path)).float()
        else:  # .pt
            mel = torch.load(mel_path)
        
        # Ensure correct shape (n_mels, time)
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        
        return {
            "audio": torch.zeros(1),  # Placeholder
            "mel": mel,
            "mel_path": str(mel_path),
            "file_name": file_name,
        }
