import os
import sys
from pathlib import Path
from typing import Optional
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torchaudio
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.transforms import MelSpectrogram, MelSpectrogramConfig
from src.model import Generator
from src.datasets import CustomDirDataset


def load_generator(checkpoint_path: str, config: DictConfig, device: str) -> Generator:
    """Load trained generator from checkpoint."""
    generator = instantiate(config.model.generator)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "generator" in checkpoint:
            generator.load_state_dict(checkpoint["generator"])
        else:
            generator.load_state_dict(checkpoint)
    else:
        generator.load_state_dict(checkpoint)
    
    generator = generator.to(device)
    generator.eval()
    
    # Remove weight normalization for faster inference
    generator.remove_weight_norm()
    
    return generator


@torch.no_grad()
def synthesize(
    generator: Generator,
    dataset: CustomDirDataset,
    output_dir: Path,
    device: str,
    sample_rate: int = 22050,
) -> None:
    """
    Synthesize audio from mel-spectrograms.
    
    Args:
        generator: Trained HiFi-GAN generator
        dataset: Dataset with mel-spectrograms
        output_dir: Directory to save synthesized audio
        device: Device to run inference on
        sample_rate: Output audio sample rate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    total_audio_length = 0
    
    for i in tqdm(range(len(dataset)), desc="Synthesizing"):
        sample = dataset[i]
        mel = sample["mel"].unsqueeze(0).to(device)  # Add batch dimension
        file_name = sample.get("file_name", f"audio_{i}")
        
        # Measure synthesis time
        start_time = time.time()
        
        # Generate audio
        audio_gen = generator(mel)
        
        synthesis_time = time.time() - start_time
        
        # Remove batch and channel dimensions
        audio_gen = audio_gen.squeeze().cpu()
        
        # Normalize audio
        if audio_gen.abs().max() > 0:
            audio_gen = audio_gen / audio_gen.abs().max() * 0.95
        
        # Save audio
        output_path = output_dir / f"{file_name}.wav"
        torchaudio.save(
            str(output_path),
            audio_gen.unsqueeze(0),
            sample_rate,
        )
        
        # Track timing statistics
        audio_length = len(audio_gen) / sample_rate
        total_time += synthesis_time
        total_audio_length += audio_length
    
    # Print statistics
    rtf = total_time / total_audio_length if total_audio_length > 0 else 0
    speed_factor = 1 / rtf if rtf > 0 else float("inf")
    
    print(f"\nSynthesis complete!")
    print(f"Total audio generated: {total_audio_length:.2f} seconds")
    print(f"Total synthesis time: {total_time:.2f} seconds")
    print(f"Real-time factor: {rtf:.4f}")
    print(f"Speed: {speed_factor:.2f}x faster than real-time")
    print(f"Output saved to: {output_dir}")


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config: DictConfig):
    """Main synthesis function."""
    print("=" * 60)
    print("HiFi-GAN Audio Synthesis")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60)
    
    # Set device
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create mel-spectrogram transform
    mel_config = MelSpectrogramConfig(
        sr=config.mel.sr,
        n_fft=config.mel.n_fft,
        win_length=config.mel.win_length,
        hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels,
        f_min=config.mel.f_min,
        f_max=config.mel.f_max,
    )
    mel_spec = MelSpectrogram(mel_config)
    
    # Create dataset
    print(f"Loading audio from: {config.input_dir}")
    dataset = CustomDirDataset(
        audio_dir=config.input_dir,
        mel_spec_transform=mel_spec,
        target_sr=config.mel.sr,
        resynthesize=config.resynthesize,
    )
    
    print(f"Found {len(dataset)} audio files")
    
    # Load generator
    print(f"Loading checkpoint: {config.checkpoint}")
    generator = load_generator(config.checkpoint, config, device)
    print(f"Generator loaded successfully")
    
    # Synthesize audio
    output_dir = Path(config.output_dir)
    synthesize(
        generator=generator,
        dataset=dataset,
        output_dir=output_dir,
        device=device,
        sample_rate=config.mel.sr,
    )


if __name__ == "__main__":
    main()
