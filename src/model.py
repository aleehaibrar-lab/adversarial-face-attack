import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedModel(nn.Module):
    """
    Wraps a backbone with an internal ImageNet Normalize step.

    Why this exists: FGSMAttack (attack.py) expects images in raw [0, 1]
    pixel space -- it perturbs and clamps with `torch.clamp(x, 0, 1)`.
    But ImageNet-pretrained backbones expect normalized input. Previously
    utils.load_image() normalized the image BEFORE the attack, which put
    it far outside [0, 1] (roughly [-2.1, 2.7]) -- so attack.py's clamp
    was silently destroying most of the image on every attack step.

    Keeping images in [0, 1] end-to-end and normalizing only right
    before the backbone (here) fixes that mismatch.
    """

    def __init__(self, backbone):
        super().__init__()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(self.normalize(x))


def load_pretrained_resnet50(pretrained=True, device='cpu'):
    """Load pre-trained ResNet50, wrapped so it accepts raw [0,1] images."""
    backbone = models.resnet50(pretrained=pretrained)
    model = NormalizedModel(backbone)
    model.to(device)
    model.eval()
    return model


def load_inception_v3(pretrained=True, device='cpu'):
    """Load pre-trained Inception V3, wrapped so it accepts raw [0,1] images."""
    backbone = models.inception_v3(pretrained=pretrained)
    model = NormalizedModel(backbone)
    model.to(device)
    model.eval()
    return model


def extract_features(model, image_tensor, device='cpu'):
    """Extract features from image (image_tensor expected in [0,1])"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = model(image_tensor)
    return features