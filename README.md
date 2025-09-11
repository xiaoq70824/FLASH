# FLASH: A Unified Frequency-Domain Framework for All-Weather Image Restoration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17098611.svg)](https://doi.org/10.5281/zenodo.17098611)

> 🔗 **Related Publication**: This code is directly related to our manuscript submitted to *The Visual Computer* journal.
> 
> ⚠️ **Please cite our paper if you use this code.**

---

## 🎯 Abstract
Adverse weather conditions significantly degrade image quality, affecting visual perception and downstream tasks. Traditional methods often employ fixed frequency decomposition patterns, failing to capture distinct frequency characteristics of different weather degradations. 

We introduce **FLASH**, a frequency-domain learning network that achieves unified restoration through two core innovations: the **Frequency-Driven Histogram Attention (FDHA)** mechanism and the **No-Activation Frequency Block (NAFB)**. FDHA revolutionizes attention computation by utilizing frequency domain magnitude distributions, while NAFB serves as a lightweight backbone eliminating traditional activation functions. 

Experimental results on three benchmark datasets demonstrate FLASH's superiority, achieving **33.21dB average PSNR** with a **2.31dB improvement** over baselines, while maintaining computational efficiency with only **13M parameters**.

---

## ✨ Key Features

### 🚀 Core Innovations

#### Frequency-Driven Histogram Attention (FDHA)
<div align="center">
<img width="400" height="800" alt="model_2" src="https://github.com/user-attachments/assets/6476828e-cc5d-4473-86e4-200bacc3aa66" />
<p><em>Frequency-Driven Histogram Attention mechanism</em></p>
</div>

- 🎯 Replaces pixel intensity grouping with frequency domain magnitude distributions
- 🔄 Dual-branch processing: FHR for local features, BHR for global dependencies  
- 🎨 Enables precise modeling of different weather degradation patterns

#### No-Activation Frequency Block (NAFB)
<div align="center">
<img width="400" height="500" alt="model_1" src="https://github.com/user-attachments/assets/9bb40e78-61b3-464c-8bac-3d481b2292a9" />
<p><em>No-Activation Frequency Block design</em></p>
</div>

- ⚡ Eliminates traditional activation functions for improved efficiency
- 🔗 Integrates spatial and frequency domain processing seamlessly

---

## 🏗️ Overall Architecture
<div align="center">
<img width="800" height="448" alt="figure2" src="https://github.com/user-attachments/assets/1591f07a-1a5d-4567-8f89-da0c8ba860dc" />
</div>

The figure presents the overall architecture of the proposed FLASH network for adverse weather image restoration tasks, illustrating three key components:

### 🔧 Architecture Components

**🏛️ Complete FLASH Architecture (a)**: Employs a U-shaped encoder-decoder design with multiscale feature processing capabilities. The network utilizes NAFB blocks as fundamental building units and integrates FDHA modules at each scale level for frequency domain-guided feature enhancement.

**🎯 Frequency-Driven Histogram Attention (b)**: Demonstrates the core innovation of frequency domain-guided feature sorting and dual-branch attention computation through FFT-based magnitude sorting, followed by parallel BHR and FHR attention calculations.

**⚡ No-Activation Frequency Block (c)**: Presents a lightweight design that eliminates traditional activation functions while maintaining robust feature representation capabilities through depthwise separable convolution and the SimpleGate mechanism.

---

## 🚀 Getting Started

### 📦 Installation
```bash
git clone https://github.com/xiaoq70824/FLASH.git
cd FLASH
pip install -r requirements.txt
```

## 🎓 Training

### Multi-GPU Support

FLASH supports distributed training with configurable hyperparameters in get_args_parser() method in train.py.

### Dataset Setup
Organize your training and validation datasets as follows:
```bash
Your_Dataset/
├── train/
│   ├── input/     # Degraded weather images
│   └── gt/        # Corresponding ground truth clean images
└── val/
    ├── input/     # Validation degraded images  
    └── gt/        # Validation ground truth images
```

### Configuration
Update dataset paths in get_args_parser():
```bash
parser.add_argument('--train_dataset_path', default='/your/path/to/train/', type=str)
parser.add_argument('--val_dataset_path', default='/your/path/to/val/', type=str)
```

### Start Training
```bash
python train.py
```

💡 The training script automatically handles distributed training, checkpointing, loss logging, and metrics calculation.

---

## Inference
## Pretrained Weights
The pretrained model weights are available in the weights/ folder:
weights/last.pth - Main pretrained model for all-weather image restoration
## Configuration Setup
Before running inference, update the paths in test_config dictionary in test.py:
Required Paths:
checkpoint_dir: Path to weights folder (default: ./weights)
checkpoint_file: Weight filename (default: last.pth)
checkpoint_file: Weight filename (default: last.pth)
output_dir: Path where restored images will be saved
Example Configuration:
```bash
test_config = {
    'checkpoint_dir': './weights',
    'checkpoint_file': 'last.pth',
    'test_data_dir': '/path/to/your/test/images',
    'output_dir': './results',
    # ... other parameters
}
```
## Input/Output Structure
```bash
test_images/          # Your degraded images
├── image1.jpg
├── image2.png
└── ...

results/              # Restored images (auto-created)
├── image1.jpg
├── image2.png
└── ...
```
## Supported Formats
Input: .jpg, .png, .bmp, .tiff
The script automatically handles image resizing and padding for optimal restoration
## Run Inference
After configuring all paths properly, simply run:
```bash
python test.py
```
The model will process all weather degradations (rain, snow, haze) automatically in a single pass.

## Results
### Performance Comparison
FLASH achieves state-of-the-art performance across three benchmark datasets for adverse weather image restoration. Our method demonstrates consistent superiority over both specialized single-degradation approaches and unified multi-weather restoration methods.

| Method | Snow100K (PSNR/SSIM) | Outdoor-rain (PSNR/SSIM) | RESIDE (PSNR/SSIM) | Average (PSNR/SSIM) |
|--------|----------------------|---------------------------|-------------------|-------------------|
| **Specialized Methods** | | | | |
| URDRN | - | 29.62/0.9151 | - | 29.62/0.9151 |
| FSADNet | - | - | 29.96/0.9685 | 29.96/0.9685 |
| LMQFormer | 28.41/0.8762 | - | - | 28.41/0.8762 |
| **Unified Methods** | | | | |
| FocalNet | 30.90/0.9118 | 27.12/0.8858 | 32.16/0.9811 | 30.06/0.9263 |
| WGWSNet | 31.03/0.9122 | 27.10/0.8961 | 32.64/0.9841 | 30.26/0.9308 |
| TANet | 31.06/0.9112 | 27.92/0.8850 | 33.57/0.9822 | 30.85/0.9261 |
| NAFNet | 30.85/0.9104 | 28.06/0.8836 | 33.79/0.9826 | 30.90/0.9255 |
| **FLASH (Ours)** | **32.53/0.9284** | **30.06/0.9184** | **37.03/0.9893** | **33.21/0.9454** |

## Key Achievements
33.21dB average PSNR with 2.31dB improvement over the best baseline method

0.9454 average SSIM, demonstrating superior structural similarity preservation

Lightweight architecture with only 13M parameters

Unified framework handling multiple weather degradations (snow, rain, haze) simultaneously

## Citation
If you use this code, please cite our paper:
```bibtex
@article{zhang2024flash,
  title={FLASH: A Unified Frequency-Domain Framework for All-Weather Image Restoration},
  author={Zhang, Xinsheng and Feng, Siyuan},
  journal={The Visual Computer},
  year={2024},
  note={Under review}
}
```
