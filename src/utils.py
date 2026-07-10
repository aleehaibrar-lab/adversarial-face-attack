import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    tensor = tensor.squeeze(0)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array([0.229, 0.224, 0.225]) + 
           np.array([0.485, 0.456, 0.406])) * 255
    return Image.fromarray(img.astype(np.uint8))

def save_results(original, adversarial, perturbation, target, output_dir):
    """Save comparison images"""
    os.makedirs(output_dir, exist_ok=True)
    
    original_img = tensor_to_image(original)
    adversarial_img = tensor_to_image(adversarial)
    perturbation_img = tensor_to_image(perturbation)
    target_img = tensor_to_image(target)
    
    original_img.save(os.path.join(output_dir, '01_original.jpg'))
    adversarial_img.save(os.path.join(output_dir, '02_adversarial.jpg'))
    perturbation_img.save(os.path.join(output_dir, '03_perturbation.jpg'))
    target_img.save(os.path.join(output_dir, '04_target.jpg'))
