import torch
import torchaudio
from pathlib import Path

# Test sentences for MOS evaluation (from homework assignment)
TEST_SENTENCES = [
    "Привет, это тест системы синтеза речи.",
    "Нейронные сети генерируют реалистичную речь.",
    "Качество звука постоянно улучшается.",
    "Добрый день, как ваши дела?",
    "Сегодня прекрасная погода для прогулки.",
]

# Output directory - adjust this path as needed
PROJECT_ROOT = Path(__file__).parent.absolute()
OUTPUT_DIR = PROJECT_ROOT / "data" / "test" / "audio"
TARGET_SAMPLE_RATE = 22050  # Same as training config


def main():
    print("Creating test audio files using Silero TTS")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load Silero TTS model
    print("\nLoading Silero TTS model...")
    device = torch.device('cpu')
    
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v3_1_ru'
    )
    model.to(device)
    print("Model loaded successfully")
    
    # Silero v3_1_ru sample rate
    silero_sample_rate = 48000
    
    # Available speakers: 'aidar', 'baya', 'kseniya', 'xenia', 'eugene'
    speaker = 'baya'
    print(f"Using speaker: {speaker}")
    
    # Generate audio for each sentence
    print("\nGenerating audio files...")
    for i, text in enumerate(TEST_SENTENCES):
        print(f"\n[{i+1}/{len(TEST_SENTENCES)}] {text}")
        
        # Generate audio with Silero
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=silero_sample_rate
        )
        
        # Ensure audio is 2D: (1, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Resample to target sample rate (22050 Hz)
        if silero_sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=silero_sample_rate,
                new_freq=TARGET_SAMPLE_RATE
            )
            audio = resampler(audio)
        
        # Normalize audio
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max() * 0.95
        
        # Save audio file
        output_path = OUTPUT_DIR / f"test_sentence_{i+1:02d}.wav"
        torchaudio.save(str(output_path), audio, TARGET_SAMPLE_RATE)
        print(f"  Saved: {output_path.name}")
    
    print("\n" + "=" * 60)
    print("All audio files created successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Run analysis.ipynb section 2 (External Data Analysis)")
    print("2. Listen to generated audio in analysis_output/")
    print("3. Fill in the answers in the notebook")


if __name__ == "__main__":
    main()
