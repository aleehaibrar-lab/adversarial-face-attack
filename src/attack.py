import torch
import torchvision.models as models

def load_pretrained_resnet50(pretrained=True, device='cpu'):
    """Load pre-trained ResNet50"""
    model = models.resnet50(pretrained=pretrained)
    model.to(device)
    model.eval()
    return model

def load_inception_v3(pretrained=True, device='cpu'):
    """Load pre-trained Inception V3"""
    model = models.inception_v3(pretrained=pretrained)
    model.to(device)
    model.eval()
    return model

def extract_features(model, image_tensor, device='cpu'):
    """Extract features from image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = model(image_tensor)
    return features
