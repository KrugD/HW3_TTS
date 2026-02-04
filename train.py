import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.transforms import MelSpectrogram, MelSpectrogramConfig
from src.model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.loss import HiFiGANLoss
from src.datasets import RuslanDataset
from src.datasets.collate import TrainingCollator, InferenceCollator
from src.trainer import Trainer
from src.logger import CometMLWriter
from src.utils import set_random_seed, save_checkpoint, load_checkpoint


def setup_dataloaders(config, mel_spec):
    """Create training and validation dataloaders."""
    # Training dataset
    train_dataset = RuslanDataset(
        data_dir=config.datasets.train.data_dir,
        split="train",
        mel_spec_transform=mel_spec,
        target_sr=config.datasets.train.target_sr,
        segment_size=config.datasets.train.segment_size,
        val_ratio=config.datasets.train.val_ratio,
        limit=config.datasets.train.limit,
    )
    
    # Validation dataset
    val_dataset = RuslanDataset(
        data_dir=config.datasets.val.data_dir,
        split="val",
        mel_spec_transform=mel_spec,
        target_sr=config.datasets.val.target_sr,
        segment_size=config.datasets.val.segment_size,
        val_ratio=config.datasets.val.val_ratio,
        limit=config.datasets.val.limit,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.train.batch_size,
        shuffle=config.dataloader.train.shuffle,
        num_workers=config.dataloader.train.num_workers,
        pin_memory=config.dataloader.train.pin_memory,
        drop_last=config.dataloader.train.drop_last,
        collate_fn=train_dataset.collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataloader.val.batch_size,
        shuffle=config.dataloader.val.shuffle,
        num_workers=config.dataloader.val.num_workers,
        pin_memory=config.dataloader.val.pin_memory,
        drop_last=config.dataloader.val.drop_last,
        collate_fn=val_dataset.collate_fn,
    )
    
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config: DictConfig):
    """Main training function."""
    # Load environment variables (for COMET_API_KEY)
    load_dotenv()
    
    # Print configuration
    print("=" * 60)
    print("HiFi-GAN Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60)
    
    # Set random seed
    set_random_seed(config.seed)
    
    # Set device
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create mel-spectrogram config
    mel_config = MelSpectrogramConfig(
        sr=config.mel.sr,
        n_fft=config.mel.n_fft,
        win_length=config.mel.win_length,
        hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels,
        f_min=config.mel.f_min,
        f_max=config.mel.f_max,
    )
    
    # Create separate mel transforms for dataset (CPU) and trainer (GPU)
    mel_spec_cpu = MelSpectrogram(mel_config)  # For dataset workers
    mel_spec_gpu = MelSpectrogram(mel_config)  # For trainer (will be moved to device)
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = setup_dataloaders(config, mel_spec_cpu)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create models
    print("Creating models...")
    generator = instantiate(config.model.generator)
    mpd = instantiate(config.model.mpd)
    msd = instantiate(config.model.msd)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"MPD parameters: {sum(p.numel() for p in mpd.parameters()):,}")
    print(f"MSD parameters: {sum(p.numel() for p in msd.parameters()):,}")
    
    # Create loss function
    loss_fn = instantiate(config.loss)
    
    # Create optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config.optimizer.generator.lr,
        betas=tuple(config.optimizer.generator.betas),
        weight_decay=config.optimizer.generator.weight_decay,
    )
    
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=config.optimizer.discriminator.lr,
        betas=tuple(config.optimizer.discriminator.betas),
        weight_decay=config.optimizer.discriminator.weight_decay,
    )
    
    # Create schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.scheduler.gamma
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.scheduler.gamma
    )
    
    # Create logger
    logger = CometMLWriter(
        project_name=config.writer.project_name,
        experiment_name=config.trainer.run_name,
        log_audio=config.writer.log_audio,
        log_spectrograms=config.writer.log_spectrograms,
        sample_rate=config.writer.sample_rate,
        disabled=config.writer.disabled,
    )
    
    # Log hyperparameters
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    
    # Create trainer
    trainer = Trainer(
        generator=generator,
        mpd=mpd,
        msd=msd,
        loss_fn=loss_fn,
        mel_spec=mel_spec_gpu,  # Use GPU mel transform for trainer
        optim_g=optim_g,
        optim_d=optim_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        config=OmegaConf.to_container(config.trainer, resolve=True),
        device=device,
        save_dir=config.trainer.save_dir,
    )
    
    # Resume from checkpoint if specified
    if config.resume_checkpoint is not None:
        trainer.load_checkpoint(config.resume_checkpoint)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
