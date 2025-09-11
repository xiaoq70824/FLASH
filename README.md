# FLASH
FLASH: A Unified Frequency-Domain Framework for All-Weather Image Restoration

🔗 **Related Publication**: This code is directly related to our manuscript submitted to *The Visual Computer* journal.

⚠️ **Please cite our paper if you use this code.**

## Abstract
Adverse weather conditions significantly degrade image quality, affecting visual perception and downstream tasks. Traditional methods often employ fixed frequency decomposition patterns, failing to capture distinct frequency characteristics of different weather degradations. We introduce FLASH, a frequency-domain learning network that achieves unified restoration through two core innovations: the Frequency-Driven Histogram Attention (FDHA) mechanism and the No-Activation Frequency Block (NAFB). FDHA revolutionizes attention computation by utilizing frequency domain magnitude distributions, while NAFB serves as a lightweight backbone eliminating traditional activation functions. Experimental results on three benchmark datasets demonstrate FLASH's superiority, achieving 33.21dB average PSNR with a 2.31dB improvement over baselines, while maintaining computational efficiency with only 13M parameters.

## Installation
```bash
git clone https://github.com/xiaoq70824/FLASH.git
cd FLASH
pip install -r requirements.txt
'''

## Training

### Distributed Training Support
FLASH supports multi-GPU distributed training. All training hyperparameters are defined in the `get_args_parser()` method in `train.py`. If you need to adjust any settings (learning rate, batch size, number of GPUs, etc.), modify the corresponding parameters in that method.

### Quick Start
After setting appropriate hyperparameters, simply run:
```bash
python train.py
