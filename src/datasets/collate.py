from typing import List, Dict, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def collate_fn(
    batch: List[Dict[str, Any]],
    segment_size: int = 8192,
    training: bool = True,
) -> Dict[str, Tensor]:
    """
    Collate function for HiFi-GAN training.
    
    During training, random segments of fixed length are cropped from audio
    and corresponding mel-spectrograms. During inference, full sequences are used.
    
    Args:
        batch: List of samples, each containing:
            - audio: Audio waveform tensor (time,)
            - mel: Mel-spectrogram tensor (n_mels, mel_time)
            - audio_path: Path to audio file (optional)
        segment_size: Length of audio segments for training
        training: Whether in training mode (crop segments) or inference mode (full)
        
    Returns:
        Dictionary containing:
            - audio: Batched audio tensor (batch, 1, time)
            - mel: Batched mel-spectrogram tensor (batch, n_mels, mel_time)
            - audio_paths: List of audio paths (if available)
    """
    if training:
        return _collate_training(batch, segment_size)
    else:
        return _collate_inference(batch)


def _collate_training(
    batch: List[Dict[str, Any]],
    segment_size: int = 8192,
) -> Dict[str, Tensor]:
    """
    Collate function for training - crops random segments.
    
    Args:
        batch: List of samples
        segment_size: Length of audio segments
        
    Returns:
        Batched tensors
    """
    hop_length = 256  # HiFi-GAN hop length
    mel_segment_size = segment_size // hop_length
    
    audios = []
    mels = []
    audio_paths = []
    
    for sample in batch:
        audio = sample["audio"]
        mel = sample["mel"]
        
        # Ensure audio is long enough
        if audio.shape[0] < segment_size:
            # Pad audio if too short
            padding = segment_size - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
            # Recompute mel for padded audio
            mel_len = audio.shape[0] // hop_length
            if mel.shape[1] < mel_len:
                mel = torch.nn.functional.pad(mel, (0, mel_len - mel.shape[1]))
        
        # Random crop
        max_audio_start = audio.shape[0] - segment_size
        audio_start = torch.randint(0, max_audio_start + 1, (1,)).item() if max_audio_start > 0 else 0
        audio_segment = audio[audio_start:audio_start + segment_size]
        
        mel_start = audio_start // hop_length
        mel_segment = mel[:, mel_start:mel_start + mel_segment_size]
        
        # Ensure mel has correct size
        if mel_segment.shape[1] < mel_segment_size:
            mel_segment = torch.nn.functional.pad(
                mel_segment, (0, mel_segment_size - mel_segment.shape[1])
            )
        
        audios.append(audio_segment.unsqueeze(0))  # Add channel dimension
        mels.append(mel_segment)
        
        if "audio_path" in sample:
            audio_paths.append(sample["audio_path"])
    
    result = {
        "audio": torch.stack(audios),  # (batch, 1, segment_size)
        "mel": torch.stack(mels),  # (batch, n_mels, mel_segment_size)
    }
    
    if audio_paths:
        result["audio_paths"] = audio_paths
    
    return result


def _collate_inference(batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """
    Collate function for inference - uses full sequences with padding.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors with padding
    """
    audios = []
    mels = []
    audio_paths = []
    audio_lengths = []
    mel_lengths = []
    
    for sample in batch:
        audio = sample["audio"]
        mel = sample["mel"]
        
        audios.append(audio)
        mels.append(mel.transpose(0, 1))  # (mel_time, n_mels) for padding
        audio_lengths.append(audio.shape[0])
        mel_lengths.append(mel.shape[1])
        
        if "audio_path" in sample:
            audio_paths.append(sample["audio_path"])
    
    # Pad sequences
    audios_padded = pad_sequence(audios, batch_first=True)  # (batch, max_time)
    mels_padded = pad_sequence(mels, batch_first=True)  # (batch, max_mel_time, n_mels)
    mels_padded = mels_padded.transpose(1, 2)  # (batch, n_mels, max_mel_time)
    
    result = {
        "audio": audios_padded.unsqueeze(1),  # (batch, 1, max_time)
        "mel": mels_padded,  # (batch, n_mels, max_mel_time)
        "audio_lengths": torch.tensor(audio_lengths),
        "mel_lengths": torch.tensor(mel_lengths),
    }
    
    if audio_paths:
        result["audio_paths"] = audio_paths
    
    return result


class TrainingCollator:
    """Collator class for training with configurable segment size."""
    
    def __init__(self, segment_size: int = 8192):
        self.segment_size = segment_size
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        return collate_fn(batch, segment_size=self.segment_size, training=True)


class InferenceCollator:
    """Collator class for inference."""
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        return collate_fn(batch, training=False)
