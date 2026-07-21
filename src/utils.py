import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess image.

    Returns a tensor in raw [0, 1] pixel space. ImageNet normalization
    is applied inside the model (see model.NormalizedModel) instead of
    here, so this stays compatible with FGSMAttack's assumption that
    images live in [0, 1] and get clamped there after each attack step.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    transform = transforms.Compose([
        transforms.ToTensor(),  # -> [0, 1], shape (C, H, W)
    ])
    return transform(img).unsqueeze(0)

def tensor_to_image(tensor):
    """Convert a [0,1]-range tensor back to a PIL Image."""
    tensor = tensor.squeeze(0)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1) * 255
    return Image.fromarray(img.astype(np.uint8))

def save_results(original, adversarial, perturbation, target, output_dir):
    """Save comparison images"""
    os.makedirs(output_dir, exist_ok=True)

    original_img = tensor_to_image(original)
    adversarial_img = tensor_to_image(adversarial)
    # Perturbation itself isn't a [0,1] image -- rescale it to be visible
    # rather than clipping most of it to black.
    pert = perturbation.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pert_vis = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
    perturbation_img = Image.fromarray((pert_vis * 255).astype(np.uint8))
    target_img = tensor_to_image(target)

    original_img.save(os.path.join(output_dir, '01_original.jpg'))
    adversarial_img.save(os.path.join(output_dir, '02_adversarial.jpg'))
    perturbation_img.save(os.path.join(output_dir, '03_perturbation.jpg'))
    target_img.save(os.path.join(output_dir, '04_target.jpg'))