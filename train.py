import argparse
import numpy as np
import os

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
import data as Data
from glob import glob
import torch.distributed as dist
import time
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from losses import ImageEnhancementLoss

from model.FLASH import FLASH

import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Import PSNR and SSIM calculation functions
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def get_args_parser():
    parser = argparse.ArgumentParser('FLASH Training', add_help=False)

    # Learning rate and optimization settings
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the model')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs for training')

    # Model parameters
    parser.add_argument('--in_channels', default=3, type=int, help='Number of input image channels')
    parser.add_argument('--out_channels', default=3, type=int, help='Number of output image channels')
    parser.add_argument('--dropout_prob', default=0, type=float, help='Dropout probability')
    parser.add_argument('--act', default='gelu', type=str, help='Activation function to use')
    parser.add_argument('--norm', default='instance', type=str, help='Normalization type')
    parser.add_argument('--dim', default=64, type=int, help='Feature dimension')

    # Training settings
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='Gradient clipping max norm')
    parser.add_argument('--val_freq', default=1, type=int, help='Validation frequency during training (in steps)')

    # Dataset settings
    # parser.add_argument('--dataset_path', default='D:/pycharm_workspace/diff_restorer/allweather_rename', type=str, help='Path to the dataset')
    parser.add_argument('--train_dataset_path', default='/root/FLASH/New_Dataset/train/', type=str,
                        help='Path to the training dataset')
    parser.add_argument('--val_dataset_path', default='/root/FLASH/New_Dataset/val', type=str,
                        help='Path to the validation dataset')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader workers')

    # Logging settings
    # parser.add_argument('--log_dir', default='D:/pycharm_workspace/diff_restorer/logs', type=str,
    #                     help='Directory for saving logs and checkpoints')
    # parser.add_argument('--checkpoint_dir', default='D:/pycharm_workspace/diff_restorer/weights', type=str,
    #                     help='Directory for saving model checkpoints')
    parser.add_argument('--log_dir', default='./root/FLASH/logs', type=str,
                        help='Directory for saving logs and checkpoints')
    parser.add_argument('--checkpoint_dir', default='/root/autodl-tmp/weights', type=str,
                        help='Directory for saving model checkpoints')
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to latest checkpoint (default: None)')
    parser.add_argument('--lr_log_dir', default='/root/autodl-tmp/lr_logs', type=str,
                        help='Directory for saving learning rate logs')

    # Device settings
    parser.add_argument('--device', default='cuda', type=str, help='Device to use for training/testing')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')

    # Distributed training parameters
    parser.add_argument('--local_rank', type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument('--n_gpus', type=int, default=4, help="Total number of GPUs for distributed training.")
    parser.add_argument('--output_dir', type=str, default='/root/FLASH/', help="Directory to save the output.")

    return parser


def setup_logging_and_save_file(args):
    local_rank = args.local_rank
    loss_log_path = os.path.join(args.output_dir, 'loss_log.txt')

    if local_rank == 0:
        # If resuming training, append; otherwise overwrite
        mode = 'a' if args.resume else 'w'
        with open(loss_log_path, mode) as f:
            if mode == 'w':  # Only write header for new training
                f.write("epoch,"
                        "train_total_loss,train_rec_loss,train_ssim_loss,"
                        "val_total_loss,val_rec_loss,val_ssim_loss\n")
    return loss_log_path


# Setting up PSNR and SSIM metrics log file
def setup_metrics_logging(args):
    """
    Set up log file for PSNR and SSIM metrics
    """
    local_rank = args.local_rank
    metrics_log_path = os.path.join(args.output_dir, 'metrics_log.txt')

    if local_rank == 0:
        # If resuming training, append; otherwise overwrite
        mode = 'a' if args.resume else 'w'
        with open(metrics_log_path, mode) as f:
            if mode == 'w':  # Only write header for new training
                f.write("epoch,val_psnr,val_ssim\n")
    return metrics_log_path


def set_random_seed(seed):
    """
    Set random seed to ensure experiment reproducibility.
    :param seed: Random seed to set
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If there are multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def initialize_model_and_optimizer(opt, resume=None):
    """Initialize model and optimizer"""
    local_rank = opt['local_rank']
    start_epoch = 0

    model = FLASH(
        inp_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        drop_out_rate=0.
    ).to(local_rank)

    optimizer = AdamW(
        model.parameters(),
        lr=opt['lr'],
        betas=(0.5, 0.999),
        weight_decay=opt['weight_decay']
    )

    # Warmup epochs
    warmup_epochs = 5

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt['epochs'],
        eta_min=5e-6
    )

    # If checkpoint path is provided, load state
    if resume is not None and os.path.isfile(resume):
        if local_rank == 0:
            print(f'Loading checkpoint from: {resume}')

        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(resume, map_location=map_location)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

        if local_rank == 0:
            print(f'Loaded checkpoint at epoch {start_epoch}')

    return model, optimizer, scheduler, start_epoch


class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, mode='min'):
        """
        Early stopping to prevent overfitting and unnecessary training

        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): 'min' or 'max' depending on whether we want to minimize or maximize the monitored value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'min':
            # If current value is smaller than best value (considering minimum improvement threshold), update best value
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:
            # If current value is larger than best value (considering minimum improvement threshold), update best value
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop


def plot_learning_rate(opt):
    """Plot learning rate change curve"""
    lr_log_path = os.path.join(opt['lr_log_dir'], 'lr_rate.txt')
    if not os.path.exists(lr_log_path):
        return

    epochs = []
    learning_rates = []

    with open(lr_log_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            epoch, lr = line.strip().split(',')
            epochs.append(int(epoch))
            learning_rates.append(float(lr))

    # Clear current figure
    plt.clf()

    # Create new figure
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, '-o', markersize=4)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.yscale('log')  # Use logarithmic scale to show learning rate changes
    plt.tight_layout()

    # Save figure - save to same file each time to achieve overwriting
    save_path = os.path.join(opt['lr_log_dir'], 'lr_schedule.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to release memory

    print(f"Learning rate curve updated to: {save_path}")


# Plotting PSNR and SSIM metrics
def plot_metrics(metrics_log_path):
    """
    Plot PSNR and SSIM curves over epochs
    """
    if not os.path.exists(metrics_log_path):
        print("Metrics log file does not exist yet, skipping plot generation.")
        return

    # Initialize data containers
    epochs = []
    psnr_values = []
    ssim_values = []

    # Read metrics log file
    try:
        with open(metrics_log_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:  # Only header or empty file
                print("Metrics log file is empty or contains only header, skipping plot generation.")
                return

            header = lines[0]  # Skip header
            for line in lines[1:]:
                try:
                    parts = line.strip().split(',')
                    epoch_num = int(parts[0])
                    psnr = float(parts[1])
                    ssim = float(parts[2])

                    epochs.append(epoch_num)
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Skipping malformed line in metrics log: {line.strip()}")
                    continue
    except Exception as e:
        print(f"Error reading metrics log file: {e}")
        return

    # If no data, return directly
    if not epochs:
        print("No valid metrics data found, skipping plot generation.")
        return

    # Create plot directory
    plot_dir = os.path.dirname(metrics_log_path)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot PSNR curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, psnr_values, 'o-', label='PSNR', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'psnr_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot SSIM curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ssim_values, 'o-', label='SSIM', linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'ssim_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PSNR and SSIM charts saved to {plot_dir}")


def plot_and_save_losses(epoch, loss_log_path):
    """
    Plot and save loss curves based on loss log file.
    Add empty data check to avoid errors on first epoch.
    """
    # Check if file exists and is not empty
    if not os.path.exists(loss_log_path):
        print("Loss log file does not exist yet, skipping plot generation.")
        return

    # Initialize data containers
    data = {
        'epochs': [],
        'train': {
            'total': [], 'rec': [], 'ssim': []
        },
        'val': {
            'total': [], 'rec': [], 'ssim': []
        }
    }

    # Read loss log file
    try:
        with open(loss_log_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:  # Only header or empty file
                print("Loss log file is empty or contains only header, skipping plot generation.")
                return

            header = lines[0]  # Skip header
            for line in lines[1:]:
                try:
                    parts = line.strip().split(',')
                    epoch_num = int(parts[0])
                    values = [float(v) for v in parts[1:]]

                    data['epochs'].append(epoch_num)

                    # Training losses
                    data['train']['total'].append(values[0])
                    data['train']['rec'].append(values[1])
                    data['train']['ssim'].append(values[2])

                    # Validation losses - modify indices to match new loss format
                    data['val']['total'].append(values[3])
                    data['val']['rec'].append(values[4])
                    data['val']['ssim'].append(values[5])
                except (IndexError, ValueError) as e:
                    print(f"Warning: Skipping malformed line in loss log: {line.strip()}")
                    continue
    except Exception as e:
        print(f"Error reading loss log file: {e}")
        return

    # If no data, return directly
    if not data['epochs']:
        print("No valid loss data found, skipping plot generation.")
        return

    # Create plot directory
    plot_dir = os.path.dirname(loss_log_path)
    os.makedirs(plot_dir, exist_ok=True)

    # Define loss types and corresponding titles
    loss_types = ['total', 'rec', 'ssim']
    titles = {
        'total': 'Total Loss',
        'rec': 'Reconstruction Loss',
        'ssim': 'SSIM Loss'
    }

    for loss_type in loss_types:
        plt.figure(figsize=(10, 6))

        train_loss = data['train'][loss_type]
        val_loss = data['val'][loss_type]
        epochs = data['epochs']

        if len(epochs) > 1:  # At least two data points
            try:
                from scipy.signal import savgol_filter

                # Use Savitzky-Golay filter for smoothing
                window_length = min(len(epochs), 5)
                if window_length % 2 == 0:
                    window_length += 1
                poly_order = min(window_length - 1, 3)

                if len(epochs) > window_length:
                    train_loss_smooth = savgol_filter(train_loss, window_length, poly_order)
                    val_loss_smooth = savgol_filter(val_loss, window_length, poly_order)
                    plt.plot(epochs, train_loss_smooth, label=f'Train {titles[loss_type]}', linewidth=2)
                    plt.plot(epochs, val_loss_smooth, label=f'Validation {titles[loss_type]}',
                             linewidth=2, linestyle='--')
                else:
                    plt.plot(epochs, train_loss, label=f'Train {titles[loss_type]}', linewidth=2)
                    plt.plot(epochs, val_loss, label=f'Validation {titles[loss_type]}',
                             linewidth=2, linestyle='--')
            except Exception as e:
                print(f"Warning: Smoothing failed for {loss_type}, plotting raw data: {e}")
                plt.plot(epochs, train_loss, label=f'Train {titles[loss_type]}', linewidth=2)
                plt.plot(epochs, val_loss, label=f'Validation {titles[loss_type]}',
                         linewidth=2, linestyle='--')
        else:
            # Single data point, just plot points
            plt.scatter(epochs, train_loss, label=f'Train {titles[loss_type]}')
            plt.scatter(epochs, val_loss, label=f'Validation {titles[loss_type]}')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{titles[loss_type]} over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Set appropriate y-axis range
        if len(epochs) > 0:
            all_losses = train_loss + val_loss
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            margin = (max_loss - min_loss) * 0.05
            plt.ylim(min_loss - margin, max_loss + margin)

        # Use integer ticks
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()

        # Save figure
        plot_path = os.path.join(plot_dir, f'{loss_type}_loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"All loss plots saved to {plot_dir}")


def load_datasets(opt):
    """
    Load training and validation datasets and create corresponding data loaders.
    Note: Training set uses distributed loading, validation set only loads on main process
    """
    train_dataset_path = opt['train_dataset_path']
    val_dataset_path = opt['val_dataset_path']
    batch_size = opt['batch_size']
    num_workers = opt['num_workers']
    local_rank = opt['local_rank']
    n_gpus = opt['n_gpus']

    # Load training and validation set image paths
    train_input_files = sorted(glob(os.path.join(train_dataset_path, 'input', '*.*')))
    train_gt_files = sorted(glob(os.path.join(train_dataset_path, 'gt', '*.*')))

    # Create training dataset object
    train_dataset = Data.AllWeatherDataset(
        root_dir=train_dataset_path,
        input_list=train_input_files,
        gt_list=train_gt_files
    )

    # Training set uses distributed sampler
    train_loader = Data.create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        rank=local_rank,
        world_size=n_gpus
    )

    # Validation set only loads on main process (rank 0)
    if local_rank == 0:
        val_input_files = sorted(glob(os.path.join(val_dataset_path, 'input', '*.*')))
        val_gt_files = sorted(glob(os.path.join(val_dataset_path, 'gt', '*.*')))

        val_dataset = Data.AllWeatherDataset(
            root_dir=val_dataset_path,
            input_list=val_input_files,
            gt_list=val_gt_files
        )

        # Validation set uses normal DataLoader, not distributed sampling
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        val_loader = None

    return train_loader, val_loader


def validate(val_loader, model, criterion, opt):
    """
    Validation process, calculate average loss on validation set as well as PSNR and SSIM.
    Note: Only run validation on main process, then broadcast results to all processes
    """
    model.eval()
    local_rank = opt['local_rank']

    # Initialize validation loss values and metrics
    unweighted_val_total_loss = 0.0
    weighted_val_total_loss = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0

    # Only run validation on main process (rank 0)
    if local_rank == 0 and val_loader is not None:
        criterion.val_loss_accumulator = {'rec': 0.0, 'ssim': 0.0}
        criterion.num_batches = 0

        # Initialize PSNR and SSIM accumulators
        total_psnr = 0.0
        total_ssim = 0.0
        total_images = 0

        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(val_loader):
                    input_images = batch_data['input_image'].to(local_rank)
                    gt_images = batch_data['gt_image'].to(local_rank)

                    # Forward propagation - test mode
                    restored_image = model(input_images)

                    # Total loss
                    total_loss = criterion(
                        enhanced_image=restored_image,
                        target_image=gt_images,
                        is_validation=True
                    )

                    # Get current accumulated losses
                    current_val_accumulated = criterion.val_loss_accumulator
                    avg_reconstruction_loss = current_val_accumulated['rec'] / (batch_idx + 1)
                    avg_ssim_loss = current_val_accumulated['ssim'] / (batch_idx + 1)

                    # Calculate PSNR and SSIM
                    batch_psnr = 0.0
                    batch_ssim = 0.0

                    for i in range(restored_image.size(0)):
                        # Note: images have been converted via ToTensor and are in [0,1] range
                        # Need to convert tensors to numpy arrays for PSNR and SSIM calculation
                        pred = restored_image[i].detach().cpu().permute(1, 2, 0).numpy()  # CHW -> HWC
                        target = gt_images[i].detach().cpu().permute(1, 2, 0).numpy()  # CHW -> HWC

                        # Ensure values are in [0,1] range
                        pred = np.clip(pred, 0, 1)
                        target = np.clip(target, 0, 1)

                        # Calculate PSNR and SSIM
                        # Data range is 1.0 because images are normalized to [0,1]
                        psnr_val = compute_psnr(target, pred, data_range=1.0)
                        ssim_val = compute_ssim(target, pred, channel_axis=-1, data_range=1.0)

                        batch_psnr += psnr_val
                        batch_ssim += ssim_val

                        # Add to total values
                        total_psnr += psnr_val
                        total_ssim += ssim_val
                        total_images += 1

                    # Calculate current batch average PSNR and SSIM
                    if restored_image.size(0) > 0:
                        batch_psnr /= restored_image.size(0)
                        batch_ssim /= restored_image.size(0)

                    # Update progress bar
                    pbar.set_postfix(
                        total_loss=total_loss.item(),
                        reconstruction_loss=avg_reconstruction_loss,
                        ssim_loss=avg_ssim_loss,
                        psnr=batch_psnr,
                        ssim=batch_ssim
                    )
                    pbar.update(1)

        # Calculate and record average losses
        criterion.update_epoch_loss(is_validation=True)

        # Get latest average validation losses
        avg_val_reconstruction_loss = criterion.val_loss_history['rec'][-1]
        avg_val_ssim_loss = criterion.val_loss_history['ssim'][-1]

        # Calculate total average loss
        unweighted_val_total_loss = (
                avg_val_reconstruction_loss +
                avg_val_ssim_loss
        )

        weighted_val_total_loss = (
                criterion.lambda_rec * avg_val_reconstruction_loss +
                criterion.lambda_ssim * avg_val_ssim_loss
        )

        # Calculate average PSNR and SSIM
        if total_images > 0:
            avg_psnr = total_psnr / total_images
            avg_ssim = total_ssim / total_images

        print(
            f"Validation Completed, Avg Total Loss: {weighted_val_total_loss:.4f}, "
            f"Avg Reconstruction Loss: {avg_val_reconstruction_loss:.4f}, "
            f"Avg SSIM Loss: {avg_val_ssim_loss:.4f}, "
            f"Avg PSNR: {avg_psnr:.4f}, "
            f"Avg SSIM: {avg_ssim:.4f}"
        )

    # Broadcast validation results to all processes
    if opt['n_gpus'] > 1:
        # Create tensors for broadcasting
        unweighted_tensor = torch.tensor([unweighted_val_total_loss], device=f"cuda:{local_rank}")
        weighted_tensor = torch.tensor([weighted_val_total_loss], device=f"cuda:{local_rank}")
        psnr_tensor = torch.tensor([avg_psnr], device=f"cuda:{local_rank}")
        ssim_tensor = torch.tensor([avg_ssim], device=f"cuda:{local_rank}")

        # Broadcast from rank 0 to all processes
        dist.broadcast(unweighted_tensor, 0)
        dist.broadcast(weighted_tensor, 0)
        dist.broadcast(psnr_tensor, 0)
        dist.broadcast(ssim_tensor, 0)

        # Update local variables
        unweighted_val_total_loss = unweighted_tensor.item()
        weighted_val_total_loss = weighted_tensor.item()
        avg_psnr = psnr_tensor.item()
        avg_ssim = ssim_tensor.item()

    return unweighted_val_total_loss, weighted_val_total_loss, avg_psnr, avg_ssim


def train(train_loader, val_loader, model, optimizer, scheduler, opt, loss_log_path, metrics_log_path=None,
          start_epoch=0):
    """
    Training code
    """

    local_rank = opt['local_rank']

    if local_rank == 0:
        # Ensure directory exists
        os.makedirs(opt['lr_log_dir'], exist_ok=True)
        lr_log_path = os.path.join(opt['lr_log_dir'], 'lr_rate.txt')

        # If resuming training, use append mode; otherwise use write mode
        mode = 'a' if opt.get('resume') else 'w'
        with open(lr_log_path, mode) as f:
            if mode == 'w':  # Only write header for new training
                f.write("epoch,learning_rate\n")

    # Initialize loss function
    criterion = ImageEnhancementLoss(device=local_rank).to(local_rank)

    early_stopping = EarlyStopping(
        patience=10,
        min_delta=5e-5,
        mode='min'
    )

    # Record training start time
    start_time = time.time()

    # Start training loop
    for epoch in range(start_epoch, opt['epochs']):
        print(f"Starting epoch {epoch + 1}/{opt['epochs']} training")
        epoch_start_time = time.time()

        model.train()

        # Call train_loader's sampler's set_epoch method at the beginning of each epoch
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
            if val_loader is not None and isinstance(val_loader.sampler,
                                                     torch.utils.data.distributed.DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

        # Reset accumulated losses for each epoch
        criterion.epoch_loss_accumulator = {'rec': 0.0, 'ssim': 0.0}
        criterion.num_batches = 0

        # Use tqdm progress bar to show progress of each batch
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{opt['epochs']}", unit="batch",
                  disable=(local_rank != 0)) as pbar:
            for batch_idx, batch_data in enumerate(train_loader):
                # Move inputs and labels to corresponding GPU
                input_images = batch_data['input_image'].to(local_rank)
                gt_images = batch_data['gt_image'].to(local_rank)

                # Clear optimizer gradients
                optimizer.zero_grad()

                restored_image = model(input_images)

                # 4. Calculate loss through ImageEnhancementLoss forward method
                total_loss = criterion(
                    enhanced_image=restored_image,
                    target_image=gt_images,
                    is_validation=False  # During training process
                )

                # Backward propagation
                total_loss.backward()

                # Gradient clipping
                if opt['clip_max_norm']:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], opt['clip_max_norm'])

                # Optimizer update
                optimizer.step()

                # Get specific loss values (calculated from epoch_loss_accumulator)
                current_losses = {
                    'rec': criterion.epoch_loss_accumulator['rec'] / criterion.num_batches,
                    'ssim': criterion.epoch_loss_accumulator['ssim'] / criterion.num_batches
                }

                # Update progress bar
                pbar.set_postfix(
                    total_loss=total_loss.item(),
                    **current_losses
                )
                pbar.update(1)

        # Calculate average loss for each epoch and update loss history
        criterion.update_epoch_loss()

        weighted_train_loss = (
                criterion.lambda_rec * criterion.loss_history['rec'][-1] +
                criterion.lambda_ssim * criterion.loss_history['ssim'][-1]
        )

        # Add synchronization point before validation to ensure all processes complete training steps
        torch.distributed.barrier()

        # Validate model and calculate validation loss, now also returns PSNR and SSIM
        unweighted_val_total_loss, weighted_val_total_loss, val_psnr, val_ssim = validate(val_loader, model, criterion,
                                                                                          opt)

        # Ensure validation results are broadcasted to all processes
        torch.distributed.barrier()

        if local_rank == 0:   # Only print in main process
            print(f"Epoch [{epoch + 1}/{opt['epochs']}] "
                  f"Training Weighted Loss: {weighted_train_loss:.4f}, "
                  f"Validation Weighted Loss: {weighted_val_total_loss:.4f}, "
                  f"Validation PSNR: {val_psnr:.4f}, "
                  f"Validation SSIM: {val_ssim:.4f}")

        # Update learning rate
        scheduler.step()

        # Record current learning rate
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr:.2e}')

            # Record learning rate to file
            with open(lr_log_path, 'a') as f:
                f.write(f"{epoch + 1},{current_lr:.8e}\n")
            try:
                plot_learning_rate(opt)
            except Exception as e:
                print(f"[warn] plot_learning_rate failed: {e}")

        # Prepare early stopping decision variable
        should_stop = False

        if local_rank == 0:  # Only check early stopping in main process
            if early_stopping(weighted_val_total_loss):
                print(
                    f"\nEarly stopping triggered! No improvement in validation loss for {early_stopping.patience} epochs.")
                # Save best model state
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss_history': criterion.loss_history,
                    'best_val_loss': early_stopping.best_value,
                    'psnr': val_psnr,
                    'ssim': val_ssim
                }
                best_model_path = os.path.join(opt['checkpoint_dir'], f'best_model_epoch_{epoch + 1}.pth')
                torch.save(checkpoint, best_model_path)
                should_stop = True

        # Broadcast early stopping decision to all processes
        stop_tensor = torch.tensor([1 if should_stop else 0], device=f"cuda:{local_rank}")
        torch.distributed.broadcast(stop_tensor, 0)
        should_stop = bool(stop_tensor.item())

        # If early stopping triggered, all processes exit training loop
        if should_stop:
            break

        # Save model weights and optimizer state
        if local_rank == 0:
            # Save model checkpoint, now includes PSNR and SSIM info in filename
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_history': criterion.loss_history,
                'psnr': val_psnr,
                'ssim': val_ssim
            }

            checkpoint_path = os.path.join(
                opt['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}_loss_{unweighted_val_total_loss:.4f}_psnr_{val_psnr:.2f}_ssim_{val_ssim:.4f}.pth'
            )
            torch.save(checkpoint, checkpoint_path)

            # Record each epoch's loss values to txt file
            with open(loss_log_path, 'a') as f:
                train_losses = criterion.loss_history
                val_losses = criterion.val_loss_history

                f.write(f"{epoch + 1},"
                        f"{sum(train_losses[k][-1] for k in train_losses.keys()):.4f},"
                        f"{train_losses['rec'][-1]:.4f},"
                        f"{train_losses['ssim'][-1]:.4f},"
                        f"{unweighted_val_total_loss:.4f},"
                        f"{val_losses['rec'][-1]:.4f},"
                        f"{val_losses['ssim'][-1]:.4f}\n")

            # Record PSNR and SSIM to metrics log file
            if metrics_log_path and local_rank == 0:
                with open(metrics_log_path, 'a') as f:
                    f.write(f"{epoch + 1},{val_psnr:.4f},{val_ssim:.4f}\n")
                # Plot PSNR and SSIM curves
                try:
                    plot_metrics(metrics_log_path)
                except Exception as e:
                    print(f"[warn] plot_metrics failed: {e}")

            # Plot and save loss curves
            if local_rank == 0:
                try:
                    plot_and_save_losses(epoch + 1, loss_log_path)
                except Exception as e:
                    print(f"[warn] plot_and_save_losses failed: {e}")

        # Calculate and print remaining estimated training time
        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (epoch + 1 - start_epoch)) * (opt['epochs'] - start_epoch)
        remaining_time = estimated_total_time - elapsed_time
        if local_rank == 0:
            print(f"Epoch {epoch + 1} completed: "
                  f"duration: {epoch_duration:.2f}s, "
                  f"estimated remaining: {remaining_time:.2f}s")


def main_worker(rank, world_size, args):
    # Set random seed
    # set_random_seed(args.seed)

    # Distributed setup: initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set path configuration
    opt = {
        'train_dataset_path': args.train_dataset_path,  # Training set path
        'val_dataset_path': args.val_dataset_path,  # Validation set path
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'path': {
            'log': args.log_dir,
            'tb_logger': os.path.join(args.log_dir, 'tb_logs')
        },
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'dropout_prob': args.dropout_prob,
        'act': args.act,
        'norm': args.norm,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'clip_max_norm': args.clip_max_norm,
        'val_freq': args.val_freq,
        'device': rank,  # Use rank as device number
        'local_rank': rank,
        'n_gpus': world_size,
        'lr_log_dir': args.lr_log_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'loss_type': 'l1',
        'output_dir': args.output_dir
    }

    # Set up logging and visualization tools, only rank 0 process is responsible for output logs
    loss_log_path = setup_logging_and_save_file(args) if rank == 0 else None

    # Set up PSNR and SSIM metrics log file
    metrics_log_path = setup_metrics_logging(args) if rank == 0 else None

    # Load training and validation datasets
    train_loader, val_loader = load_datasets(opt)

    # Initialize model and optimizer
    model, optimizer, scheduler, start_epoch = initialize_model_and_optimizer(opt, resume=args.resume)

    # Load model to corresponding device and wrap as DDP
    device = torch.device(f"cuda:{rank}")
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Synchronize all processes, ensure model initialization is complete before continuing
    dist.barrier()

    # Start training, pass metrics_log_path parameter
    train(train_loader, val_loader, model, optimizer, scheduler, opt, loss_log_path, metrics_log_path,
          start_epoch=start_epoch)

    # Destroy process group
    dist.destroy_process_group()


if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '39283'  # Choose an unused port

    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found.")

    # Start multi-process distributed training
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
