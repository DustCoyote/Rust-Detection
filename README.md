# RUST Segmentation Project

This repository contains a Python training script and data for a rust segmentation project based on a U-Net workflow.

## Project Structure

- `train_rust_segmentation_unet.py`: main training, validation, and test-image inference script.
- `dataset/content/dataset/images`: input images.
- `dataset/content/dataset/masks`: segmentation masks.
- `test/`: sample test images.


## Quick Start

1. Make sure PyTorch, torchvision, Pillow, matplotlib, and numpy are installed.
2. Run `python train_rust_segmentation_unet.py --mode all`.
