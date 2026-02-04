from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm, spectral_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


class PeriodDiscriminator(nn.Module):
    """
    Sub-discriminator for Multi-Period Discriminator.
    
    Reshapes 1D audio to 2D based on the given period and applies 2D convolutions.
    The kernel width is always 1 to process periodic samples independently.
    
    Args:
        period: The period for reshaping audio (e.g., 2, 3, 5, 7, 11)
        kernel_size: Kernel size in height dimension
        stride: Stride in height dimension
        use_spectral_norm: Whether to use spectral normalization (default: False)
    """
    
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Convolutional layers with increasing channels
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        
        # Final projection to 1 channel
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass through period discriminator.
        
        Args:
            x: Audio waveform of shape (batch, 1, time)
            
        Returns:
            output: Discriminator output
            fmap: List of feature maps from each layer
        """
        fmap = []
        
        # Get batch size and time
        b, c, t = x.shape
        
        # Pad to make time divisible by period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        # Reshape to 2D: (batch, 1, time) -> (batch, 1, time/period, period)
        x = x.view(b, c, t // self.period, self.period)
        
        # Apply convolutional layers
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        # Final projection
        x = self.conv_post(x)
        fmap.append(x)
        
        # Flatten output
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD).
    
    Consists of multiple sub-discriminators, each handling a different period
    of the input audio. Periods are set to prime numbers [2, 3, 5, 7, 11] to
    avoid overlaps.
    
    Args:
        periods: List of periods for sub-discriminators
        use_spectral_norm: Whether to use spectral normalization
    """
    
    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period, use_spectral_norm=use_spectral_norm)
            for period in periods
        ])
    
    def forward(self, y: Tensor, y_hat: Tensor) -> Tuple[
        List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]
    ]:
        """
        Forward pass through all period discriminators.
        
        Args:
            y: Real audio waveform of shape (batch, 1, time)
            y_hat: Generated audio waveform of shape (batch, 1, time)
            
        Returns:
            y_d_rs: List of real outputs from each sub-discriminator
            y_d_gs: List of generated outputs from each sub-discriminator
            fmap_rs: List of feature maps for real audio
            fmap_gs: List of feature maps for generated audio
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(nn.Module):
    """
    Sub-discriminator for Multi-Scale Discriminator.
    
    Applies 1D convolutions on the audio at a specific scale.
    Uses grouped convolutions to increase discriminator capacity.
    
    Args:
        use_spectral_norm: Whether to use spectral normalization (default: False)
    """
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
    
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass through scale discriminator.
        
        Args:
            x: Audio waveform of shape (batch, 1, time)
            
        Returns:
            output: Discriminator output
            fmap: List of feature maps from each layer
        """
        fmap = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD).
    
    Consists of three sub-discriminators operating at different scales:
    1. Raw audio
    2. 2x average-pooled audio
    3. 4x average-pooled audio
    
    The first sub-discriminator uses spectral normalization, others use weight normalization.
    
    Args:
        use_spectral_norm_first: Whether to use spectral norm for first discriminator
    """
    
    def __init__(self, use_spectral_norm_first: bool = True):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=use_spectral_norm_first),
            ScaleDiscriminator(use_spectral_norm=False),
            ScaleDiscriminator(use_spectral_norm=False),
        ])
        
        # Average pooling for downscaling
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])
    
    def forward(self, y: Tensor, y_hat: Tensor) -> Tuple[
        List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]
    ]:
        """
        Forward pass through all scale discriminators.
        
        Args:
            y: Real audio waveform of shape (batch, 1, time)
            y_hat: Generated audio waveform of shape (batch, 1, time)
            
        Returns:
            y_d_rs: List of real outputs from each sub-discriminator
            y_d_gs: List of generated outputs from each sub-discriminator
            fmap_rs: List of feature maps for real audio
            fmap_gs: List of feature maps for generated audio
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            # Apply pooling for 2nd and 3rd discriminators
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
