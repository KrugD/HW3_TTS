from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm, remove_weight_norm


def init_weights(m, mean=0.0, std=0.01):
    """Initialize weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    """
    Residual Block with dilated convolutions.
    
    Each residual block contains two sets of dilated convolutions.
    The output is the sum of input and the convolution outputs.
    
    Args:
        channels: Number of input/output channels
        kernel_size: Kernel size for convolutions
        dilations: List of dilation rates for each layer
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )
        
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Output tensor of same shape
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = conv2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class MRF(nn.Module):
    """
    Multi-Receptive Field Fusion (MRF) module.
    
    The MRF module consists of multiple residual blocks with different kernel sizes
    and dilation rates. The outputs of all residual blocks are summed to create
    diverse receptive field patterns.
    
    Args:
        channels: Number of input/output channels
        kernel_sizes: List of kernel sizes for residual blocks
        dilations: List of dilation rate tuples for each residual block
    """
    
    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        
        self.resblocks = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilations):
            self.resblocks.append(ResBlock(channels, kernel_size, dilation))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MRF module.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Output tensor of same shape (sum of all residual block outputs)
        """
        output = None
        for resblock in self.resblocks:
            if output is None:
                output = resblock(x)
            else:
                output = output + resblock(x)
        return output / len(self.resblocks)
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for resblock in self.resblocks:
            resblock.remove_weight_norm()


class Generator(nn.Module):
    """
    HiFi-GAN Generator.
    
    The generator upsamples mel-spectrograms to raw audio waveforms using
    transposed convolutions and MRF modules.
    
    Args:
        in_channels: Number of input channels (mel bands)
        hidden_channels: Hidden dimension (hu in the paper)
        upsample_rates: Upsampling rates for each transposed convolution
        upsample_kernel_sizes: Kernel sizes for transposed convolutions
        resblock_kernel_sizes: Kernel sizes for MRF residual blocks
        resblock_dilations: Dilation rates for MRF residual blocks
    
    The product of upsample_rates should equal hop_length (256 = 8 * 8 * 2 * 2).
    
    Model variants (from paper):
        V1: hu=512, ku=[16,16,4,4], kr=[3,7,11], Dr=[[1,3,5]]*3
        V2: hu=128, ku=[16,16,4,4], kr=[3,7,11], Dr=[[1,3,5]]*3
        V3: hu=256, ku=[16,16,4,4], kr=[3,5,7], Dr=[[1,2],[2,6],[3,12]]
    """
    
    def __init__(
        self,
        in_channels: int = 80,
        hidden_channels: int = 512,
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilations: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
    ):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Initial convolution to project mel-spectrogram to hidden dimension
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, hidden_channels, 7, 1, padding=3)
        )
        
        # Upsampling layers with MRF modules
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        for i, (upsample_rate, kernel_size) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            # Calculate channels for this layer
            in_ch = hidden_channels // (2 ** i)
            out_ch = hidden_channels // (2 ** (i + 1))
            
            # Transposed convolution for upsampling
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel_size,
                        upsample_rate,
                        padding=(kernel_size - upsample_rate) // 2,
                    )
                )
            )
            
            # MRF module
            self.mrfs.append(
                MRF(out_ch, resblock_kernel_sizes, resblock_dilations)
            )
        
        # Final convolution to produce 1-channel audio
        self.conv_post = weight_norm(
            nn.Conv1d(out_ch, 1, 7, 1, padding=3)
        )
        
        # Initialize weights
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Generate audio waveform from mel-spectrogram.
        
        Args:
            x: Mel-spectrogram tensor of shape (batch, n_mels, mel_length)
            
        Returns:
            Audio waveform tensor of shape (batch, 1, audio_length)
        """
        # Initial projection
        x = self.conv_pre(x)
        
        # Upsampling with MRF
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        
        # Final projection with tanh activation
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for faster inference."""
        print("Removing weight norm from generator...")
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)
    
    @property
    def receptive_field(self) -> int:
        """Calculate the receptive field of the generator."""
        # This is an approximation - actual receptive field depends on
        # the specific configuration of kernel sizes and dilations
        return sum(self.ups[0].kernel_size) * len(self.ups)


class GeneratorV1(Generator):
    """HiFi-GAN Generator V1 configuration (highest quality)."""
    
    def __init__(self, in_channels: int = 80):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=512,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        )
