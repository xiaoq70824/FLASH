import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class ImageEnhancementLoss(nn.Module):
    def __init__(self, device=None):
        """
        Initialize ImageEnhancementLoss class

        Args:
            device (torch.device): Device information
        """
        super(ImageEnhancementLoss, self).__init__()

        # Set device
        self.device = device if device else torch.device("cpu")

        # Set loss weights
        self.lambda_rec = 1.0    # L1 reconstruction loss weight
        self.lambda_ssim = 0.5   # SSIM loss weight

        # For recording loss history
        self.loss_history = {'rec': [], 'ssim': []}
        self.val_loss_history = {'rec': [], 'ssim': []}

        # Batch accumulator
        self.epoch_loss_accumulator = {'rec': 0.0, 'ssim': 0.0}
        self.val_loss_accumulator = {'rec': 0.0, 'ssim': 0.0}
        self.num_batches = 0

    def reconstruction_loss(self, enhanced_image, target_image):
        """Calculate L1 reconstruction loss"""
        return F.l1_loss(enhanced_image, target_image)

    def ssim_loss(self, enhanced_image, target_image, win_size=11):
        """Calculate SSIM loss"""
        enhanced_image = torch.clamp(enhanced_image, 0.0, 1.0)
        target_image = torch.clamp(target_image, 0.0, 1.0)
        return 1 - ssim(enhanced_image, target_image, data_range=1.0,
                      size_average=True, win_size=win_size)

    def forward(self, enhanced_image, target_image, is_validation=False):
        """
        Forward propagation to calculate total loss

        Args:
            enhanced_image (torch.Tensor): Enhanced image
            target_image (torch.Tensor): Target image
            is_validation (bool): Whether it's validation phase
        """
        self.num_batches += 1

        # Calculate various losses
        rec_loss = self.reconstruction_loss(enhanced_image, target_image)
        ssim_loss = self.ssim_loss(enhanced_image, target_image)

        # Calculate total loss
        total_loss = self.lambda_rec * rec_loss + self.lambda_ssim * ssim_loss

        # Accumulate loss values
        if is_validation:
            self.val_loss_accumulator['rec'] += rec_loss.item()
            self.val_loss_accumulator['ssim'] += ssim_loss.item()
        else:
            self.epoch_loss_accumulator['rec'] += rec_loss.item()
            self.epoch_loss_accumulator['ssim'] += ssim_loss.item()

        return total_loss

    def update_epoch_loss(self, is_validation=False):
        """Update loss history at the end of each epoch"""
        if is_validation:
            for key in self.val_loss_accumulator:
                avg_loss = self.val_loss_accumulator[key] / self.num_batches
                self.val_loss_history[key].append(avg_loss)
            self.val_loss_accumulator = {k: 0.0 for k in self.val_loss_accumulator}
        else:
            for key in self.epoch_loss_accumulator:
                avg_loss = self.epoch_loss_accumulator[key] / self.num_batches
                self.loss_history[key].append(avg_loss)
            self.epoch_loss_accumulator = {k: 0.0 for k in self.epoch_loss_accumulator}

        self.num_batches = 0