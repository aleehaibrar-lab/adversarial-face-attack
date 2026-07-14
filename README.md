# Adversarial Face Recognition Attack

## Overview
This project implements adversarial attacks on facial recognition systems using gradient-based perturbations and deep learning techniques.

## Features
- FGSM (Fast Gradient Sign Method) attack, supporting:
  - Untargeted attacks (push away from the model's current prediction)
  - Label-targeted attacks (push toward a specific class)
  - Targeted **impersonation** attacks (push a face's feature embedding toward a target face's embedding)
- ResNet50 face/image feature extractor
- LFW (Labeled Faces in the Wild) deep-funneled dataset for source/target face pairs
- Pre-trained model weights (via torchvision)
- Sample datasets and results

## Datasets
- **LFW (deep-funneled)** - primary dataset used for selecting source and target face images
- **MCS2018** - optional additional attack dataset
- **VGG_Face** - optional face recognition benchmark

## Models Explored
- ResNet50
- Inception V3

## Quick Start
```bash
# Clone repository
git clone https://github.com/aleehaibrar-lab/adversarial-face-attack.git
cd adversarial-face-attack

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/AdversarialAttack.ipynb
```

## Usage

```python
from model import load_pretrained_resnet50
from attack import FGSMAttack
from utils import load_image, save_results

model = load_pretrained_resnet50(pretrained=True, device='cpu')
attacker = FGSMAttack(model, epsilon=0.03, device='cpu')

image = load_image('data/sample_images/test_image.jpg')
target_image = load_image('data/sample_images/target_image.jpg')

# Impersonation attack: nudge `image` so the model's features on it
# resemble the model's features on `target_image`
adversarial, perturbation = attacker.attack(image, target=target_image)

save_results(image, adversarial, perturbation, target_image, 'results/')
```
