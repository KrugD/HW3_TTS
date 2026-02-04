import random

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.io_utils import ROOT_PATH


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_saving_and_logging(config):
    """Setup directories for saving checkpoints and logs."""
    save_dir = ROOT_PATH / config.trainer.save_dir / config.trainer.run_name
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    OmegaConf.save(config, save_dir / "config.yaml")
    
    return save_dir


def get_dataloaders(config, mel_spec_transform):
    """Create dataloaders for training and validation."""
    dataloaders = {}
    
    for split, split_config in config.datasets.items():
        dataset = instantiate(split_config, mel_spec_transform=mel_spec_transform)
        
        dataloader_config = config.dataloader.get(split, config.dataloader.train)
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=dataloader_config.get("shuffle", split == "train"),
            num_workers=dataloader_config.get("num_workers", 4),
            pin_memory=dataloader_config.get("pin_memory", True),
            drop_last=dataloader_config.get("drop_last", split == "train"),
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )
    
    return dataloaders


def instantiate_model(config):
    """Instantiate generator and discriminator models."""
    generator = instantiate(config.model.generator)
    mpd = instantiate(config.model.mpd)
    msd = instantiate(config.model.msd)
    
    return generator, mpd, msd


def instantiate_loss(config):
    """Instantiate loss functions."""
    return instantiate(config.loss)


def instantiate_optimizer(config, generator, mpd, msd):
    """Instantiate optimizers for generator and discriminators."""
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
    
    return optim_g, optim_d


def instantiate_scheduler(config, optim_g, optim_d, last_epoch=-1):
    """Instantiate learning rate schedulers."""
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=config.scheduler.gamma,
        last_epoch=last_epoch,
    )
    
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=config.scheduler.gamma,
        last_epoch=last_epoch,
    )
    
    return scheduler_g, scheduler_d


def load_checkpoint(checkpoint_path, generator, mpd, msd, optim_g=None, optim_d=None, 
                    scheduler_g=None, scheduler_d=None, device="cpu"):
    """Load checkpoint from path."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    generator.load_state_dict(checkpoint["generator"])
    mpd.load_state_dict(checkpoint["mpd"])
    msd.load_state_dict(checkpoint["msd"])
    
    if optim_g is not None and "optim_g" in checkpoint:
        optim_g.load_state_dict(checkpoint["optim_g"])
    if optim_d is not None and "optim_d" in checkpoint:
        optim_d.load_state_dict(checkpoint["optim_d"])
    if scheduler_g is not None and "scheduler_g" in checkpoint:
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    if scheduler_d is not None and "scheduler_d" in checkpoint:
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
    
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    
    return epoch, step, best_val_loss


def save_checkpoint(
    save_path, 
    generator, 
    mpd, 
    msd, 
    optim_g, 
    optim_d, 
    scheduler_g, 
    scheduler_d, 
    epoch, 
    step,
    best_val_loss=float("inf"),
):
    """Save checkpoint to path."""
    torch.save(
        {
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
        },
        save_path,
    )
