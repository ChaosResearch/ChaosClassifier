# Classifying conservative chaos and invariant tori by deeplearning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the official PyTorch implementation of Chaos Classification.

## Installation

### Requirements

- Linux with Python ≥ 3.6
- PyTorch >= 1.8.1
- timm >= 0.3.2
- CUDA 11.1
- An NVIDIA GPU

### Conda environment setup

```bash
conda create -n Chaos_Classification python=3.9
conda activate Chaos_Classification

# Install Pytorch and TorchVision
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install scikit-learn
pip install scikit-learn
```

## Evaluation
To evaluate a pre-trained model on test dataset with GPUs run:
```
python test.py
```

If you use this code for a paper please cite:

```
It will be updated after the paper is accepted.
```
