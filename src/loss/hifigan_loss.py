from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def generator_loss(disc_outputs: List[Tensor]) -> Tensor:
    """
    Generator adversarial loss (LS-GAN).
    
    The generator tries to make the discriminator output 1 for generated samples.
    
    L_Adv(G) = E[(D(G(s)) - 1)^2]
    
    Args:
        disc_outputs: List of discriminator outputs for generated samples
        
    Returns:
        Generator adversarial loss
    """
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((dg - 1) ** 2)
    return loss


def discriminator_loss(
    disc_real_outputs: List[Tensor],
    disc_generated_outputs: List[Tensor],
) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
    """
    Discriminator adversarial loss (LS-GAN).
    
    The discriminator tries to output 1 for real samples and 0 for generated samples.
    
    L_Adv(D) = E[(D(x) - 1)^2 + D(G(s))^2]
    
    Args:
        disc_real_outputs: List of discriminator outputs for real samples
        disc_generated_outputs: List of discriminator outputs for generated samples
        
    Returns:
        Total discriminator loss
        List of losses for real samples
        List of losses for generated samples
    """
    loss = 0
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr - 1) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    
    return loss, r_losses, g_losses


def feature_matching_loss(
    fmap_r: List[List[Tensor]],
    fmap_g: List[List[Tensor]],
) -> Tensor:
    """
    Feature matching loss.
    
    L1 distance between features of discriminator for real and generated samples.
    
    L_FM(G) = E[sum_i (1/N_i) * ||D_i(x) - D_i(G(s))||_1]
    
    Args:
        fmap_r: List of feature maps for real samples
        fmap_g: List of feature maps for generated samples
        
    Returns:
        Feature matching loss
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss


def mel_spectrogram_loss(
    mel_real: Tensor,
    mel_generated: Tensor,
) -> Tensor:
    """
    Mel-spectrogram loss.
    
    L1 distance between mel-spectrograms of real and generated audio.
    
    L_Mel(G) = E[||φ(x) - φ(G(s))||_1]
    
    Args:
        mel_real: Mel-spectrogram of real audio
        mel_generated: Mel-spectrogram of generated audio
        
    Returns:
        Mel-spectrogram loss
    """
    return F.l1_loss(mel_real, mel_generated)


class HiFiGANLoss(nn.Module):
    """
    Combined HiFi-GAN loss module.
    
    Computes generator and discriminator losses including:
    - Adversarial loss (LS-GAN)
    - Feature matching loss
    - Mel-spectrogram loss
    
    Args:
        lambda_fm: Weight for feature matching loss (default: 2)
        lambda_mel: Weight for mel-spectrogram loss (default: 45)
    """
    
    def __init__(self, lambda_fm: float = 2.0, lambda_mel: float = 45.0):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
    
    def generator_loss(
        self,
        mpd_outputs: Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]],
        msd_outputs: Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]],
        mel_real: Tensor,
        mel_generated: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        Compute total generator loss.
        
        L_G = L_Adv(G; MPD) + L_Adv(G; MSD) + 
              λ_fm * (L_FM(G; MPD) + L_FM(G; MSD)) + 
              λ_mel * L_Mel(G)
        
        Args:
            mpd_outputs: Outputs from Multi-Period Discriminator
            msd_outputs: Outputs from Multi-Scale Discriminator
            mel_real: Mel-spectrogram of real audio
            mel_generated: Mel-spectrogram of generated audio
            
        Returns:
            Total generator loss
            Dictionary with individual loss components
        """
        _, y_dg_mpd, fmap_r_mpd, fmap_g_mpd = mpd_outputs
        _, y_dg_msd, fmap_r_msd, fmap_g_msd = msd_outputs
        
        # Adversarial loss
        loss_gen_mpd = generator_loss(y_dg_mpd)
        loss_gen_msd = generator_loss(y_dg_msd)
        loss_gen = loss_gen_mpd + loss_gen_msd
        
        # Feature matching loss
        loss_fm_mpd = feature_matching_loss(fmap_r_mpd, fmap_g_mpd)
        loss_fm_msd = feature_matching_loss(fmap_r_msd, fmap_g_msd)
        loss_fm = loss_fm_mpd + loss_fm_msd
        
        # Mel-spectrogram loss
        loss_mel = mel_spectrogram_loss(mel_real, mel_generated)
        
        # Total generator loss
        loss_total = loss_gen + self.lambda_fm * loss_fm + self.lambda_mel * loss_mel
        
        loss_dict = {
            "loss_gen": loss_gen.item(),
            "loss_gen_mpd": loss_gen_mpd.item(),
            "loss_gen_msd": loss_gen_msd.item(),
            "loss_fm": loss_fm.item(),
            "loss_fm_mpd": loss_fm_mpd.item(),
            "loss_fm_msd": loss_fm_msd.item(),
            "loss_mel": loss_mel.item(),
            "loss_gen_total": loss_total.item(),
        }
        
        return loss_total, loss_dict
    
    def discriminator_loss(
        self,
        mpd_outputs: Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]],
        msd_outputs: Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]],
    ) -> Tuple[Tensor, dict]:
        """
        Compute total discriminator loss.
        
        L_D = L_Adv(D_MPD; G) + L_Adv(D_MSD; G)
        
        Args:
            mpd_outputs: Outputs from Multi-Period Discriminator
            msd_outputs: Outputs from Multi-Scale Discriminator
            
        Returns:
            Total discriminator loss
            Dictionary with individual loss components
        """
        y_dr_mpd, y_dg_mpd, _, _ = mpd_outputs
        y_dr_msd, y_dg_msd, _, _ = msd_outputs
        
        # MPD loss
        loss_disc_mpd, losses_disc_r_mpd, losses_disc_g_mpd = discriminator_loss(
            y_dr_mpd, y_dg_mpd
        )
        
        # MSD loss
        loss_disc_msd, losses_disc_r_msd, losses_disc_g_msd = discriminator_loss(
            y_dr_msd, y_dg_msd
        )
        
        # Total discriminator loss
        loss_total = loss_disc_mpd + loss_disc_msd
        
        loss_dict = {
            "loss_disc": loss_total.item(),
            "loss_disc_mpd": loss_disc_mpd.item(),
            "loss_disc_msd": loss_disc_msd.item(),
        }
        
        return loss_total, loss_dict
    
    def forward(
        self,
        mpd_outputs: Tuple,
        msd_outputs: Tuple,
        mel_real: Tensor,
        mel_generated: Tensor,
    ) -> Tuple[Tensor, Tensor, dict]:
        """
        Compute both generator and discriminator losses.
        
        Args:
            mpd_outputs: Outputs from Multi-Period Discriminator
            msd_outputs: Outputs from Multi-Scale Discriminator
            mel_real: Mel-spectrogram of real audio
            mel_generated: Mel-spectrogram of generated audio
            
        Returns:
            Generator loss
            Discriminator loss
            Dictionary with all loss components
        """
        loss_gen, gen_loss_dict = self.generator_loss(
            mpd_outputs, msd_outputs, mel_real, mel_generated
        )
        loss_disc, disc_loss_dict = self.discriminator_loss(mpd_outputs, msd_outputs)
        
        loss_dict = {**gen_loss_dict, **disc_loss_dict}
        
        return loss_gen, loss_disc, loss_dict
