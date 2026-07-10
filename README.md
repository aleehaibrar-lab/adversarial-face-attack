# Adversarial Face Recognition Attack

## Overview
This project implements adversarial attacks on facial recognition systems using spatial constraints and deep learning techniques.

## Features
- FGSM attack with spatial masking
- ResNet50 face recognition model
- DLib facial landmark detection
- Pre-trained model weights
- Sample datasets and results

## Datasets
- **MCS2018** - Main attack dataset
- **VGG_Face** - Face recognition benchmark
- **ImageNet** - General image dataset

## Models Explored
- ResNet50
- Inception V3
- Convolutional Neural Network (CNN)

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/adversarial-face-attack.git
cd adversarial-face-attack

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/AdversarialAttack.ipynb
