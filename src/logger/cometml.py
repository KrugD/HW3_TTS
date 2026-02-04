import os
from typing import Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues on Windows
import matplotlib.pyplot as plt
from torch import Tensor


from comet_ml import Experiment
COMET_AVAILABLE = True


class CometMLWriter:
    """
    CometML experiment logger.
    
    Args:
        project_name: CometML project name
        experiment_name: Name of the experiment
        api_key: CometML API key (can also be set via COMET_API_KEY env var)
        workspace: CometML workspace
        log_audio: Whether to log audio samples
        log_spectrograms: Whether to log spectrograms
        sample_rate: Audio sample rate for logging
    """
    
    def __init__(
        self,
        project_name: str = "hifigan-tts",
        experiment_name: Optional[str] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        log_audio: bool = True,
        log_spectrograms: bool = True,
        sample_rate: int = 22050,
        disabled: bool = False,
    ):
        self._log_audio_enabled = log_audio
        self._log_spectrograms_enabled = log_spectrograms
        self.sample_rate = sample_rate
        self.disabled = disabled or not COMET_AVAILABLE
        
        if self.disabled:
            self.experiment = None
            return
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("COMET_API_KEY")
        
        if api_key is None:
            print("Warning: COMET_API_KEY not set. CometML logging disabled.")
            self.disabled = True
            self.experiment = None
            return
        
        try:
            self.experiment = Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                auto_metric_logging=False,
                auto_param_logging=False,
            )
            
            if experiment_name is not None:
                self.experiment.set_name(experiment_name)
            
            print(f"CometML experiment: {self.experiment.url}")
            
        except Exception as e:
            print(f"Warning: Failed to initialize CometML: {e}")
            self.disabled = True
            self.experiment = None
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.disabled:
            return
        self.experiment.log_parameters(params)
    
    def log_metric(self, name: str, value: float, step: int):
        """Log a single metric."""
        if self.disabled:
            return
        self.experiment.log_metric(name, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log multiple metrics."""
        if self.disabled:
            return
        for name, value in metrics.items():
            self.experiment.log_metric(name, value, step=step)
    
    def log_audio(
        self,
        audio: Tensor,
        name: str,
        step: int,
        sample_rate: Optional[int] = None,
    ):
        """
        Log audio sample to CometML.
        
        Args:
            audio: Audio tensor of shape (time,) or (1, time)
            name: Name for the audio sample
            step: Training step
            sample_rate: Audio sample rate
        """
        if self.disabled or not self._log_audio_enabled:
            return
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure audio is 1D numpy array
        if isinstance(audio, Tensor):
            audio = audio.detach().cpu().numpy()
        
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Normalize to prevent clipping
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Save to temporary file and log
        try:
            import tempfile
            import soundfile as sf
            
            # Create temp file, close it, write to it, then delete
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # Close file descriptor immediately
            
            sf.write(temp_path, audio, sample_rate)
            self.experiment.log_audio(
                temp_path,
                file_name=f"{name}_step{step}.wav",
                step=step,
            )
            
            # Try to delete, but don't fail if it doesn't work on Windows
            try:
                os.unlink(temp_path)
            except PermissionError:
                pass  # Windows may still have the file locked
        except Exception as e:
            print(f"Warning: Failed to log audio {name}: {e}")
    
    def log_spectrogram(
        self,
        spectrogram: Tensor,
        name: str,
        step: int,
    ):
        """
        Log spectrogram image to CometML.
        
        Args:
            spectrogram: Spectrogram tensor of shape (freq, time) or (1, freq, time)
            name: Name for the spectrogram
            step: Training step
        """
        if self.disabled or not self._log_spectrograms_enabled:
            return
        
        # Convert to numpy
        if isinstance(spectrogram, Tensor):
            spectrogram = spectrogram.detach().cpu().numpy()
        
        if spectrogram.ndim > 2:
            spectrogram = spectrogram.squeeze()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            interpolation="none",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Frequency")
        ax.set_title(name)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        try:
            self.experiment.log_figure(
                figure_name=f"{name}_step{step}",
                figure=fig,
                step=step,
            )
        except Exception as e:
            print(f"Warning: Failed to log spectrogram {name}: {e}")
        finally:
            plt.close(fig)
    
    def log_waveform(
        self,
        waveform: Tensor,
        name: str,
        step: int,
    ):
        """
        Log waveform plot to CometML.
        
        Args:
            waveform: Waveform tensor
            name: Name for the plot
            step: Training step
        """
        if self.disabled:
            return
        
        if isinstance(waveform, Tensor):
            waveform = waveform.detach().cpu().numpy()
        
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(waveform[:min(len(waveform), 22050)])  # Plot first second
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.set_title(name)
        plt.tight_layout()
        
        try:
            self.experiment.log_figure(
                figure_name=f"{name}_step{step}",
                figure=fig,
                step=step,
            )
        except Exception as e:
            print(f"Warning: Failed to log waveform {name}: {e}")
        finally:
            plt.close(fig)
    
    def log_model(self, path: str, name: Optional[str] = None):
        """Log model checkpoint."""
        if self.disabled:
            return
        try:
            self.experiment.log_model(name or "model", str(path))
        except Exception as e:
            print(f"Warning: Failed to log model: {e}")
    
    def log_code(self, folder: str = "src"):
        """Log source code."""
        if self.disabled:
            return
        try:
            self.experiment.log_code(folder=folder)
        except Exception as e:
            print(f"Warning: Failed to log code: {e}")
    
    def end(self):
        """End the experiment."""
        if self.disabled or self.experiment is None:
            return
        self.experiment.end()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
