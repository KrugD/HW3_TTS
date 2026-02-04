from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.loss import HiFiGANLoss
from src.transforms import MelSpectrogram
from src.logger import CometMLWriter
from src.utils import save_checkpoint, load_checkpoint


class Trainer:
    """
    HiFi-GAN Trainer.
    
    Args:
        generator: HiFi-GAN generator model
        mpd: Multi-Period Discriminator
        msd: Multi-Scale Discriminator
        loss_fn: HiFi-GAN loss function
        mel_spec: Mel-spectrogram transform
        optim_g: Generator optimizer
        optim_d: Discriminator optimizer
        scheduler_g: Generator learning rate scheduler
        scheduler_d: Discriminator learning rate scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        logger: CometML logger
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        generator: Generator,
        mpd: MultiPeriodDiscriminator,
        msd: MultiScaleDiscriminator,
        loss_fn: HiFiGANLoss,
        mel_spec: MelSpectrogram,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler._LRScheduler,
        scheduler_d: torch.optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        logger: CometMLWriter,
        config: Dict[str, Any],
        device: str = "cuda",
        save_dir: str = "checkpoints",
    ):
        self.generator = generator.to(device)
        self.mpd = mpd.to(device)
        self.msd = msd.to(device)
        self.loss_fn = loss_fn
        self.mel_spec = mel_spec.to(device)
        
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.config = config
        self.device = device
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.step = 0
        
        # Config parameters
        self.n_epochs = config.get("n_epochs", 100)
        self.log_interval = config.get("log_interval", 100)
        self.save_interval = config.get("save_interval", 10000)
        self.val_interval = config.get("val_interval", 5000)
        self.log_audio_interval = config.get("log_audio_interval", 1000)
        self.grad_clip = config.get("grad_clip", None)
        
        # Early stopping parameters
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
    
    def train(self):
        """Run the full training loop."""
        print(f"Starting training for {self.n_epochs} epochs...")
        print(f"Early stopping patience: {self.early_stopping_patience} epochs")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"MPD parameters: {sum(p.numel() for p in self.mpd.parameters()):,}")
        print(f"MSD parameters: {sum(p.numel() for p in self.msd.parameters()):,}")
        
        for epoch in range(self.epoch, self.n_epochs):
            self.epoch = epoch
            self._train_epoch()
            
            # Run validation at end of epoch and check for improvement
            if self.val_loader is not None:
                val_loss = self._validate_epoch()
                
                # Check if this is the best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_best_checkpoint()
                    print(f"New best model! Val Mel Loss: {val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    print(f"No improvement for {self.epochs_without_improvement} epoch(s). "
                          f"Best: {self.best_val_loss:.4f}, Current: {val_loss:.4f}")
                
                # Early stopping check
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {self.early_stopping_patience} epochs without improvement!")
                    print(f"Best validation mel loss: {self.best_val_loss:.4f}")
                    break
            
            # Step schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Log learning rates
            self.logger.log_metric("lr_g", self.scheduler_g.get_last_lr()[0], self.step)
            self.logger.log_metric("lr_d", self.scheduler_d.get_last_lr()[0], self.step)
        
        # Save final checkpoint
        self._save_checkpoint("checkpoint_final.pt")
        print("Training completed!")
        print(f"Best validation mel loss: {self.best_val_loss:.4f}")
        self.logger.end()
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.generator.train()
        self.mpd.train()
        self.msd.train()
        
        epoch_losses = {"loss_gen_total": 0, "loss_disc": 0, "loss_mel": 0}
        n_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.n_epochs}",
            leave=True,
        )
        
        for batch in pbar:
            losses = self._train_step(batch)
            
            # Accumulate losses
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "G": f"{losses['loss_gen_total']:.3f}",
                "D": f"{losses['loss_disc']:.3f}",
                "Mel": f"{losses['loss_mel']:.3f}",
            })
            
            self.step += 1
            
            # Logging
            if self.step % self.log_interval == 0:
                self.logger.log_metrics(losses, self.step)
            
            # Log audio samples
            if self.step % self.log_audio_interval == 0:
                self._log_audio_samples(batch)
            
            # Validation
            if self.val_loader is not None and self.step % self.val_interval == 0:
                self._validate()
            
            # Save checkpoint
            if self.step % self.save_interval == 0:
                self._save_checkpoint(f"checkpoint_{self.step}.pt")
        
        # Epoch summary
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        print(f"Epoch {self.epoch + 1} - "
              f"G: {epoch_losses['loss_gen_total']:.4f}, "
              f"D: {epoch_losses['loss_disc']:.4f}, "
              f"Mel: {epoch_losses['loss_mel']:.4f}")
    
    def _train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data containing audio and mel-spectrograms
            
        Returns:
            Dictionary of loss values
        """
        audio = batch["audio"].to(self.device)  # (B, 1, T)
        mel = batch["mel"].to(self.device)  # (B, n_mels, mel_T)
        
        # Generate audio
        audio_gen = self.generator(mel)  # (B, 1, T')
        
        # Ensure same length
        min_len = min(audio.shape[-1], audio_gen.shape[-1])
        audio = audio[..., :min_len]
        audio_gen = audio_gen[..., :min_len]
        
        # ============ Discriminator Step ============
        self.optim_d.zero_grad()
        
        # Get discriminator outputs
        mpd_outputs = self.mpd(audio, audio_gen.detach())
        msd_outputs = self.msd(audio, audio_gen.detach())
        
        # Compute discriminator loss
        loss_disc, disc_loss_dict = self.loss_fn.discriminator_loss(
            mpd_outputs, msd_outputs
        )
        
        loss_disc.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.mpd.parameters()) + list(self.msd.parameters()),
                self.grad_clip
            )
        
        self.optim_d.step()
        
        # ============ Generator Step ============
        self.optim_g.zero_grad()
        
        # Get discriminator outputs for generator training
        mpd_outputs = self.mpd(audio, audio_gen)
        msd_outputs = self.msd(audio, audio_gen)
        
        # Compute mel-spectrogram of generated audio
        mel_gen = self.mel_spec(audio_gen.squeeze(1))
        
        # Ensure same mel length
        min_mel_len = min(mel.shape[-1], mel_gen.shape[-1])
        mel_input = mel[..., :min_mel_len]
        mel_gen = mel_gen[..., :min_mel_len]
        
        # Compute generator loss
        loss_gen, gen_loss_dict = self.loss_fn.generator_loss(
            mpd_outputs, msd_outputs, mel_input, mel_gen
        )
        
        loss_gen.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
        
        self.optim_g.step()
        
        # Combine loss dictionaries
        loss_dict = {**gen_loss_dict, **disc_loss_dict}
        
        return loss_dict
    
    @torch.no_grad()
    def _log_audio_samples(self, batch: Dict[str, Tensor]):
        """Log audio samples to CometML."""
        self.generator.eval()
        
        audio = batch["audio"].to(self.device)
        mel = batch["mel"].to(self.device)
        
        # Generate audio
        audio_gen = self.generator(mel)
        
        # Log first sample
        self.logger.log_audio(audio[0], "ground_truth", self.step)
        self.logger.log_audio(audio_gen[0], "generated", self.step)
        
        # Log spectrograms
        self.logger.log_spectrogram(mel[0], "mel_input", self.step)
        
        mel_gen = self.mel_spec(audio_gen[0])
        self.logger.log_spectrogram(mel_gen, "mel_generated", self.step)
        
        self.generator.train()
    
    @torch.no_grad()
    def _validate(self):
        """Run validation during training (for step-based validation)."""
        val_loss = self._validate_epoch()
        return val_loss
    
    @torch.no_grad()
    def _validate_epoch(self) -> float:
        """Run full validation and return mel loss for early stopping."""
        self.generator.eval()
        self.mpd.eval()
        self.msd.eval()
        
        val_losses = {"val_loss_gen": 0, "val_loss_disc": 0, "val_loss_mel": 0}
        n_batches = 0
        
        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            mel = batch["mel"].to(self.device)
            
            # Generate audio
            audio_gen = self.generator(mel)
            
            # Ensure same length
            min_len = min(audio.shape[-1], audio_gen.shape[-1])
            audio = audio[..., :min_len]
            audio_gen = audio_gen[..., :min_len]
            
            # Discriminator outputs
            mpd_outputs = self.mpd(audio, audio_gen)
            msd_outputs = self.msd(audio, audio_gen)
            
            # Mel-spectrogram of generated audio
            mel_gen = self.mel_spec(audio_gen.squeeze(1))
            min_mel_len = min(mel.shape[-1], mel_gen.shape[-1])
            mel_input = mel[..., :min_mel_len]
            mel_gen = mel_gen[..., :min_mel_len]
            
            # Compute losses
            loss_gen, gen_dict = self.loss_fn.generator_loss(
                mpd_outputs, msd_outputs, mel_input, mel_gen
            )
            loss_disc, disc_dict = self.loss_fn.discriminator_loss(
                mpd_outputs, msd_outputs
            )
            
            val_losses["val_loss_gen"] += gen_dict["loss_gen_total"]
            val_losses["val_loss_disc"] += disc_dict["loss_disc"]
            val_losses["val_loss_mel"] += gen_dict["loss_mel"]
            n_batches += 1
        
        # Average losses
        for k in val_losses:
            val_losses[k] /= max(n_batches, 1)
        
        # Log validation losses
        self.logger.log_metrics(val_losses, self.step)
        self.logger.log_metric("best_val_mel_loss", self.best_val_loss, self.step)
        
        print(f"Validation - G: {val_losses['val_loss_gen']:.4f}, "
              f"D: {val_losses['val_loss_disc']:.4f}, "
              f"Mel: {val_losses['val_loss_mel']:.4f}")
        
        self.generator.train()
        self.mpd.train()
        self.msd.train()
        
        # Return mel loss for early stopping (this is the most important metric)
        return val_losses["val_loss_mel"]
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        save_path = self.save_dir / filename
        
        save_checkpoint(
            save_path,
            self.generator,
            self.mpd,
            self.msd,
            self.optim_g,
            self.optim_d,
            self.scheduler_g,
            self.scheduler_d,
            self.epoch,
            self.step,
            self.best_val_loss,
        )
        
        print(f"Saved checkpoint: {save_path}")
        
        # Also save generator-only checkpoint for inference
        generator_path = self.save_dir / f"generator_{self.step}.pt"
        torch.save(self.generator.state_dict(), generator_path)
    
    def _save_best_checkpoint(self):
        """Save the best model checkpoint."""
        # Save full checkpoint for resuming training
        best_checkpoint_path = self.save_dir / "checkpoint_best.pt"
        save_checkpoint(
            best_checkpoint_path,
            self.generator,
            self.mpd,
            self.msd,
            self.optim_g,
            self.optim_d,
            self.scheduler_g,
            self.scheduler_d,
            self.epoch,
            self.step,
            self.best_val_loss,
        )
        
        # Save generator-only for inference
        best_generator_path = self.save_dir / "generator_best.pt"
        torch.save(self.generator.state_dict(), best_generator_path)
        
        print(f"Saved best checkpoint: {best_checkpoint_path}")
        print(f"Saved best generator: {best_generator_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.epoch, self.step, self.best_val_loss = load_checkpoint(
            checkpoint_path,
            self.generator,
            self.mpd,
            self.msd,
            self.optim_g,
            self.optim_d,
            self.scheduler_g,
            self.scheduler_d,
            self.device,
        )
        # Reset early stopping counter when loading checkpoint
        self.epochs_without_improvement = 0
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {self.epoch}, Step: {self.step}, Best Val Loss: {self.best_val_loss:.4f}")
