# RUST Segmentation Project

This repository contains notebooks and data for a rust segmentation project based on a U-Net workflow.

## Project Structure

- `Untitled-1.ipynb`: main training and inference notebook for rust segmentation.
- `rust.ipynb`: earlier experiment notebook.
- `dataset/content/dataset/images`: input images.
- `dataset/content/dataset/masks`: segmentation masks.
- `test/`: sample test images.

## Notes

- `best_unet_rust.pth` is intentionally excluded from version control because it is a large trained weight file.
- `ECCV_TransFusion/` is intentionally excluded because it was added here by mistake and is not part of this project.

## Quick Start

1. Open `Untitled-1.ipynb`.
2. Make sure PyTorch, torchvision, Pillow, OpenCV, matplotlib, and numpy are installed.
3. Run the notebook cells from top to bottom.
