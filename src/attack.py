import torch
import torch.nn as nn


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    Supports two modes:
      1. Untargeted: attack(image) - pushes the image away from its
         current prediction (maximizes loss on original class).
      2. Targeted/impersonation: attack(image, target=target_image) -
         pushes the image's feature representation TOWARD the target
         image's feature representation (useful for face impersonation
         attacks, where you want model(adversarial) to resemble
         model(target) rather than just flip a label).

    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial
    Examples" (2014) - https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model, epsilon=0.03, device='cpu'):
        """
        Args:
            model: target PyTorch model (already in eval mode). Used as
                   a feature extractor - its output (logits or embedding)
                   is what gets pushed toward/away from a reference.
            epsilon: perturbation magnitude (L-infinity bound)
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def attack(self, image, target=None, label=None):
        """
        Generate an adversarial example.

        Args:
            image: input tensor, shape (C, H, W) or (N, C, H, W),
                   values expected in [0, 1]
            target: optional target IMAGE tensor, same shape as `image`.
                    If provided, performs a targeted/impersonation attack:
                    perturbs `image` so the model's output on it moves
                    closer to the model's output on `target`.
            label: optional integer class label. If provided (and target
                   is None), performs a standard targeted/untargeted
                   classification attack using cross-entropy.
                   If both target and label are None, performs an
                   untargeted attack that maximizes loss on the model's
                   own current prediction for `image`.

        Returns:
            (adversarial_image, perturbation) - both tensors, same shape
            as the input image, adversarial_image clamped to [0, 1]
        """
        single_image = image.dim() == 3
        if single_image:
            image = image.unsqueeze(0)

        image = image.clone().detach().to(self.device)
        image.requires_grad = True

        if target is not None:
            # Impersonation attack: minimize distance between adversarial
            # image's features and the target image's features
            if target.dim() == 3:
                target = target.unsqueeze(0)
            target = target.to(self.device)

            with torch.no_grad():
                target_features = self.model(target)

            adv_features = self.model(image)
            loss = nn.functional.mse_loss(adv_features, target_features)

            self.model.zero_grad()
            loss.backward()

            # To MINIMIZE distance to target, step opposite the gradient
            perturbation = -self.epsilon * image.grad.sign()

        elif label is not None:
            # Standard targeted misclassification attack
            if not torch.is_tensor(label):
                label = torch.tensor([label])
            label = label.to(self.device)
            if label.dim() == 0:
                label = label.unsqueeze(0)

            output = self.model(image)
            loss = nn.functional.cross_entropy(output, label)

            self.model.zero_grad()
            loss.backward()

            # To push TOWARD label, step opposite the gradient
            perturbation = -self.epsilon * image.grad.sign()

        else:
            # Untargeted attack: maximize loss on model's own prediction
            output = self.model(image)
            pred_label = output.argmax(dim=1)
            loss = nn.functional.cross_entropy(output, pred_label)

            self.model.zero_grad()
            loss.backward()

            # To push AWAY from current prediction, step with the gradient
            perturbation = self.epsilon * image.grad.sign()

        adv_image = torch.clamp(image + perturbation, 0, 1).detach()
        perturbation = perturbation.detach()

        if single_image:
            adv_image = adv_image.squeeze(0)
            perturbation = perturbation.squeeze(0)

        return adv_image, perturbation

    # Kept for backward compatibility with earlier code that used
    # generate() directly with a label.
    def generate(self, image, label):
        adv_image, _ = self.attack(image, label=label)
        return adv_image