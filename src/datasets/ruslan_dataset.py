from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import torchaudio
from torch.utils.data import Dataset

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.datasets.collate import TrainingCollator


class RuslanDataset(Dataset):
    """
    RUSLAN dataset for HiFi-GAN vocoder training.
    
    Args:
        data_dir: Path to RUSLAN dataset directory
        split: Dataset split ('train' or 'val')
        mel_spec_transform: MelSpectrogram transform instance
        target_sr: Target sample rate (default: 22050)
        segment_size: Audio segment size for training (default: 8192)
        val_ratio: Ratio of validation samples (default: 0.05)
        limit: Limit number of samples (for debugging)
    """
    
    def __init__(
        self,
        data_dir: str = "data/ruslan",
        split: str = "train",
        mel_spec_transform: Optional[MelSpectrogram] = None,
        target_sr: int = 22050,
        segment_size: int = 8192,
        val_ratio: float = 0.05,
        limit: Optional[int] = None,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_sr = target_sr
        self.segment_size = segment_size
        
        # Create mel spectrogram transform if not provided
        # Always keep on CPU for dataset workers
        if mel_spec_transform is None:
            config = MelSpectrogramConfig(sr=target_sr)
            mel_spec_transform = MelSpectrogram(config)
        # Ensure transform is on CPU for multiprocessing
        self.mel_spec_transform = mel_spec_transform.cpu()
        
        # Find audio files
        self.audio_paths = self._find_audio_files()
        
        # Split into train/val
        n_samples = len(self.audio_paths)
        n_val = max(1, int(n_samples * val_ratio))
        
        if split == "train":
            self.audio_paths = self.audio_paths[n_val:]
        else:  # val
            self.audio_paths = self.audio_paths[:n_val]
        
        # Apply limit if specified
        if limit is not None:
            self.audio_paths = self.audio_paths[:limit]
        
        print(f"RuslanDataset [{split}]: {len(self.audio_paths)} samples")
        
        # Collate function - use same segment size for validation to avoid OOM
        self.collate_fn = TrainingCollator(segment_size)
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the dataset directory."""
        audio_dir = self.data_dir / "RUSLAN"
        
        if not audio_dir.exists():
            raise FileNotFoundError(
                f"RUSLAN audio directory not found at {audio_dir}. "
                "Please download the dataset first using download_ruslan.py"
            )
        
        # Filter out macOS resource fork files (starting with ._)
        audio_files = sorted([
            f for f in audio_dir.glob("*.wav") 
            if not f.name.startswith("._")
        ])
        
        if len(audio_files) == 0:
            raise FileNotFoundError(
                f"No .wav files found in {audio_dir}. "
                "Please check the dataset structure."
            )
        
        return audio_files
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - audio: Audio waveform tensor
                - mel: Mel-spectrogram tensor
                - audio_path: Path to audio file
        """
        audio_path = self.audio_paths[idx]
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)  # Remove channel dimension
        
        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        # Normalize audio to [-1, 1]
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()
        
        # Compute mel-spectrogram
        mel = self.mel_spec_transform(audio)
        
        return {
            "audio": audio,
            "mel": mel,
            "audio_path": str(audio_path),
        }


class RuslanDatasetWithMetadata(RuslanDataset):
    """
    RUSLAN dataset variant that uses metadata CSV for additional information.
    
    Args:
        data_dir: Path to RUSLAN dataset directory
        metadata_path: Path to metadata CSV file
        split: Dataset split ('train' or 'val')
        mel_spec_transform: MelSpectrogram transform instance
        target_sr: Target sample rate
        segment_size: Audio segment size for training
        val_ratio: Ratio of validation samples
        limit: Limit number of samples
    """
    
    def __init__(
        self,
        data_dir: str = "data/ruslan",
        metadata_path: Optional[str] = None,
        split: str = "train",
        mel_spec_transform: Optional[MelSpectrogram] = None,
        target_sr: int = 22050,
        segment_size: int = 8192,
        val_ratio: float = 0.05,
        limit: Optional[int] = None,
    ):
        self.metadata_path = metadata_path
        
        # Load metadata if available
        if metadata_path is not None and Path(metadata_path).exists():
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = None
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            mel_spec_transform=mel_spec_transform,
            target_sr=target_sr,
            segment_size=segment_size,
            val_ratio=val_ratio,
            limit=limit,
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with metadata."""
        sample = super().__getitem__(idx)
        
        # Add text transcription if available
        if self.metadata is not None:
            audio_name = Path(sample["audio_path"]).stem
            metadata_row = self.metadata[self.metadata["file_name"] == audio_name]
            if len(metadata_row) > 0:
                sample["text"] = metadata_row["text"].values[0]
        
        return sample
