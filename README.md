# Adversarial Face Recognition Attack

## Overview
This project implements adversarial attacks on facial recognition systems using gradient-based perturbations and deep learning techniques.

## Features
- Three adversarial attacks, all sharing the same `attack(image, target=..., label=...)` interface:
  - **FGSM** (Fast Gradient Sign Method) - single-step
  - **PGD** (Projected Gradient Descent) - iterative, stronger, with random start
  - **DeepFool** (linearized, minimal-perturbation) - adapted from classification to face-embedding verification
  - Each supports: untargeted attacks, label-targeted attacks, and targeted **impersonation** attacks (push a face's feature embedding toward a target face's embedding)
- ResNet50 face/image feature extractor
- LFW (Labeled Faces in the Wild) deep-funneled dataset for source/target face pairs
- Pre-trained model weights (via torchvision)
- Full evaluation harness (`src/evaluate.py`): genuine/impostor pair sampling, verification threshold search, clean accuracy/precision/recall/F1, and an **epsilon sweep** per attack with mean/median similarity-drop statistics saved to `results/metrics_<attack>.csv` / `.json`
- Visualization script (`src/run_visualize.py`): ASR-vs-epsilon curves, cosine-distance distribution histograms, and a qualitative perturbation-visibility grid
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

### Full evaluation (epsilon sweep, ASR, statistics)

```bash
cd src

# Run one attack across a range of epsilon (perturbation budget) values.
# Saves results/metrics_<attack>.csv and .json.
python evaluate.py --attack fgsm     --epsilons 0.01,0.03,0.05,0.1 --n-pairs 200
python evaluate.py --attack pgd      --epsilons 0.01,0.03,0.05,0.1 --n-pairs 200
python evaluate.py --attack deepfool --epsilons 0.01,0.03,0.05,0.1 --n-pairs 200
```

Each run reports, per epsilon: attack success rate, accuracy under attack,
mean/median cosine-distance before & after the attack (similarity drop),
and the average L-infinity perturbation actually used.

### Visualization

```bash
cd src

# ASR-vs-epsilon comparison across all three attacks, plus cosine-distance
# distribution histograms for one attack (reads the saved metrics files,
# no model/GPU needed):
python run_visualize.py --results-dir ../results --attacks fgsm,pgd,deepfool --attack fgsm

# Qualitative perturbation-visibility grid (needs the model + a sample pair):
python run_visualize.py --qualitative --attack fgsm \
    --epsilons 0.01,0.03,0.05,0.1 \
    --source ../data/sample_images/test_image.jpg \
    --target ../data/sample_images/target_image.jpg
```

Plots are written to `results/visualizations/`.
