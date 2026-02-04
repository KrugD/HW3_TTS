import os
import argparse
import zipfile
import tarfile
from pathlib import Path

import pandas as pd
import json

import gdown

from tqdm import tqdm


def extract_archive(archive_path: str, extract_dir: str) -> bool:
    """Extract tar.gz archive."""
    print(f"Extracting {archive_path}...")
    
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        
        # Remove archive after extraction
        os.remove(archive_path)
        print(f"Removed archive: {archive_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False


def download_ruslan(output_dir: str = "data/ruslan") -> bool:
    """
    Download and extract RUSLAN dataset from Google Drive.
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        True if successful, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Google Drive links
    dataset_url = "https://drive.google.com/uc?id=1Ye9IqnOvCjDc8NdMQol-bu-gWJGJjbB1"
    metadata_url = "https://drive.google.com/uc?id=11TD_ZwIOo-Wo75GYv-OWWOS3ABmwmAdK"
    
    tar_path = output_dir / "RUSLAN.tar.gz"
    metadata_path = output_dir / "metadata_RUSLAN_222000.csv"
    
    # Download dataset archive
    print(f"Downloading RUSLAN dataset to {tar_path}...")
    try:
        gdown.download(dataset_url, str(tar_path), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    # Extract dataset
    if tar_path.exists():
        if not extract_archive(str(tar_path), str(output_dir)):
            return False
    else:
        print(f"Archive not found: {tar_path}")
        return False
    
    # Download metadata
    print(f"Downloading metadata to {metadata_path}...")
    try:
        gdown.download(metadata_url, str(metadata_path), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        # Continue without metadata - not critical
    
    # Verify download
    ruslan_dir = output_dir / "RUSLAN"
    if ruslan_dir.exists():
        wav_files = list(ruslan_dir.glob("*.wav"))
        print(f"\nRUSLAN dataset downloaded successfully!")
        print(f"Audio directory: {ruslan_dir}")
        print(f"Number of audio files: {len(wav_files)}")
        
        if metadata_path.exists():
            print(f"Metadata file: {metadata_path}")
        
        return True
    else:
        print(f"Error: RUSLAN directory not found at {ruslan_dir}")
        return False


def create_metadata(data_dir: str) -> str:
    """
    Create metadata JSON file from audio files.
    
    Args:
        data_dir: Directory containing RUSLAN dataset
        
    Returns:
        Path to created metadata file
    """
    data_dir = Path(data_dir)
    audio_dir = data_dir / "RUSLAN"
    
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return None
    
    metadata = []
    
    # Find all audio files
    wav_files = sorted(audio_dir.glob("*.wav"))
    
    for wav_file in tqdm(wav_files, desc="Creating metadata"):
        # Check for corresponding text file
        txt_file = wav_file.with_suffix('.txt')
        
        entry = {
            "audio_path": str(wav_file.relative_to(data_dir)),
            "file_name": wav_file.stem,
        }
        
        if txt_file.exists():
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    entry["text"] = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read {txt_file}: {e}")
        
        metadata.append(entry)
    
    # Save metadata
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Created metadata with {len(metadata)} entries")
    print(f"Saved to: {metadata_path}")
    
    # Also create CSV
    df = pd.DataFrame(metadata)
    csv_path = data_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"Also saved to: {csv_path}")
    
    return str(metadata_path)


def print_dataset_info(data_dir: str):
    """Print information about the downloaded dataset."""
    data_dir = Path(data_dir)
    audio_dir = data_dir / "RUSLAN"
    
    if not audio_dir.exists():
        print("Dataset not found. Please run download first.")
        return
    
    wav_files = list(audio_dir.glob("*.wav"))
    
    print("\n" + "=" * 60)
    print("RUSLAN Dataset Information")
    print("=" * 60)
    print(f"Location: {audio_dir}")
    print(f"Number of audio files: {len(wav_files)}")
    
    # Check a sample file for info
    if wav_files:
        import torchaudio
        sample_audio, sr = torchaudio.load(wav_files[0])
        print(f"Sample rate: {sr} Hz")
        print(f"Channels: {sample_audio.shape[0]}")
        print(f"Sample duration: {sample_audio.shape[1] / sr:.2f} seconds")
    
    # Check metadata
    metadata_csv = data_dir / "metadata_RUSLAN_222000.csv"
    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv, sep="|", header=None, names=["id", "text"])
        print(f"\nMetadata entries: {len(df)}")
        print(f"Columns: {list(df.columns)}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download RUSLAN dataset for HiFi-GAN training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ruslan",
        help="Output directory for dataset (default: data/ruslan)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download and only create metadata"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_dataset_info(args.output_dir)
        return
    
    if args.skip_download:
        print("Skipping download, creating metadata only...")
        create_metadata(args.output_dir)
    else:
        # Download dataset
        success = download_ruslan(args.output_dir)
        
        if success:
            # Create additional metadata
            create_metadata(args.output_dir)
            
            # Print info
            print_dataset_info(args.output_dir)
        else:
            print("Download failed!")
            exit(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
