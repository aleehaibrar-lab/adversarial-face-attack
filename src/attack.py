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


class PGDAttack:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    PGD is FGSM applied iteratively with a small step size, projecting
    back onto the L-infinity epsilon-ball after every step. It's a
    strictly stronger (and slower) attack than single-step FGSM, and is
    generally treated as the standard "first-order adversary" baseline
    for evaluating robustness.

    Supports the same three modes as FGSMAttack: untargeted,
    label-targeted, and target-image (impersonation) attacks, so it can
    be swapped into evaluate.py's evaluation loop with no other changes.

    Reference: Madry et al., "Towards Deep Learning Models Resistant to
    Adversarial Attacks" (2018) - https://arxiv.org/abs/1706.06083
    """

    def __init__(self, model, epsilon=0.03, alpha=None, steps=10,
                 random_start=True, device='cpu'):
        """
        Args:
            model: target PyTorch model (already in eval mode).
            epsilon: L-infinity perturbation budget.
            alpha: per-step size. Defaults to epsilon / 4 (a common rule
                   of thumb) if not given.
            steps: number of PGD iterations.
            random_start: if True, initialize from a random point inside
                   the epsilon-ball instead of the clean image (helps
                   avoid gradient masking / obfuscated-gradient artifacts).
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else max(epsilon / 4, 1e-4)
        self.steps = steps
        self.random_start = random_start
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def _step_direction(self, image, target=None, label=None):
        """One gradient computation, returns the sign direction to step in."""
        image = image.clone().detach().requires_grad_(True)

        if target is not None:
            with torch.no_grad():
                target_features = self.model(target)
            adv_features = self.model(image)
            loss = nn.functional.mse_loss(adv_features, target_features)
            self.model.zero_grad()
            loss.backward()
            # minimize distance to target -> step opposite the gradient
            return -image.grad.sign()

        elif label is not None:
            output = self.model(image)
            loss = nn.functional.cross_entropy(output, label)
            self.model.zero_grad()
            loss.backward()
            return -image.grad.sign()

        else:
            output = self.model(image)
            pred_label = output.argmax(dim=1)
            loss = nn.functional.cross_entropy(output, pred_label)
            self.model.zero_grad()
            loss.backward()
            return image.grad.sign()

    def attack(self, image, target=None, label=None):
        """
        Generate an adversarial example via PGD.

        Args / Returns: same shape/semantics as FGSMAttack.attack().
        """
        single_image = image.dim() == 3
        if single_image:
            image = image.unsqueeze(0)
        image = image.clone().detach().to(self.device)

        if target is not None and target.dim() == 3:
            target = target.unsqueeze(0)
        if target is not None:
            target = target.to(self.device)

        if label is not None:
            if not torch.is_tensor(label):
                label = torch.tensor([label])
            label = label.to(self.device)
            if label.dim() == 0:
                label = label.unsqueeze(0)

        original_image = image.clone().detach()

        if self.random_start:
            noise = torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
            adv_image = torch.clamp(image + noise, 0, 1).detach()
        else:
            adv_image = image.clone().detach()

        for _ in range(self.steps):
            direction = self._step_direction(adv_image, target=target, label=label)
            adv_image = adv_image.detach() + self.alpha * direction
            # project back into the epsilon-ball around the original image
            delta = torch.clamp(adv_image - original_image, -self.epsilon, self.epsilon)
            adv_image = torch.clamp(original_image + delta, 0, 1).detach()

        perturbation = (adv_image - original_image).detach()

        if single_image:
            adv_image = adv_image.squeeze(0)
            perturbation = perturbation.squeeze(0)

        return adv_image, perturbation


class DeepFoolAttack:
    """
    DeepFool-style minimal-perturbation adversarial attack, adapted for
    embedding/feature space rather than classification logits.

    Classic DeepFool (Moosavi-Dezfooli et al., 2016) iteratively moves a
    point across the *nearest* linearized decision boundary, giving a
    much smaller perturbation than FGSM/PGD for the same success. There
    is no discrete decision boundary in embedding space, so this version
    linearizes the verification boundary instead: at each step it takes
    the gradient of the embedding distance (to the target, for
    impersonation; or to the image's own clean embedding, for the
    untargeted case) and takes the *smallest* step, in the true gradient
    direction (not just its sign, unlike FGSM/PGD), that is estimated to
    cross the similarity/distance verification threshold. This keeps
    DeepFool's core idea -- minimal perturbation via local linearization
    -- while working with a verification threshold instead of class
    labels.

    Reference: "DeepFool: a simple and accurate method to fool deep
    neural networks" - https://arxiv.org/abs/1511.04599
    """

    def __init__(self, model, threshold=0.5, max_iter=50, overshoot=0.02,
                 max_epsilon=0.3, device='cpu'):
        """
        Args:
            model: target PyTorch model (already in eval mode).
            threshold: cosine-similarity verification threshold to cross.
            max_iter: maximum number of linearization steps.
            overshoot: small multiplicative overshoot applied to each
                   step (standard DeepFool trick to actually cross the
                   boundary rather than land exactly on it).
            max_epsilon: L-infinity cap on the *total* perturbation, so
                   this attack stays comparable to FGSM/PGD at the same
                   epsilon in the sweep, and never diverges.
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.threshold = threshold
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.max_epsilon = max_epsilon
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _cosine_sim(a, b):
        a = a / a.norm(dim=1, keepdim=True)
        b = b / b.norm(dim=1, keepdim=True)
        return (a * b).sum(dim=1)

    def attack(self, image, target=None, label=None):
        """
        Generate an adversarial example via linearized minimal-step search.

        Args / Returns: same shape/semantics as FGSMAttack.attack(). Only
        the `target` (impersonation) mode and the untargeted mode are
        supported; `label`-based classification attacks are not (this
        attack is defined over the embedding, not class logits).
        """
        if label is not None:
            raise NotImplementedError(
                "DeepFoolAttack operates on embeddings; use FGSMAttack or "
                "PGDAttack for label-based classification attacks."
            )

        single_image = image.dim() == 3
        if single_image:
            image = image.unsqueeze(0)
        original_image = image.clone().detach().to(self.device)

        if target is not None:
            if target.dim() == 3:
                target = target.unsqueeze(0)
            target = target.to(self.device)
            with torch.no_grad():
                target_features = self.model(target)
        else:
            with torch.no_grad():
                target_features = self.model(original_image)

        adv_image = original_image.clone().detach()

        for _ in range(self.max_iter):
            adv_image = adv_image.clone().detach().requires_grad_(True)
            adv_features = self.model(adv_image)
            sim = self._cosine_sim(adv_features, target_features)

            if target is not None and sim.item() > self.threshold:
                break  # verifier already says "same identity" -> success
            if target is None and sim.item() < self.threshold:
                break  # pushed far enough from its own clean embedding

            # move to (target is not None: increase similarity) or
            # (target is None: decrease similarity)
            loss = sim.sum() if target is not None else -sim.sum()
            self.model.zero_grad()
            loss.backward()

            grad = adv_image.grad.detach()
            grad_norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
            # smallest step, in the *true* gradient direction, scaled by
            # how far `sim` currently is from the threshold
            step_size = (self.threshold - sim).abs() / grad_norm
            step = (1 + self.overshoot) * step_size.view(-1, 1, 1, 1) * grad

            adv_image = adv_image.detach() + step
            # cap total perturbation so this stays comparable to FGSM/PGD
            delta = torch.clamp(adv_image - original_image,
                                 -self.max_epsilon, self.max_epsilon)
            adv_image = torch.clamp(original_image + delta, 0, 1).detach()

        perturbation = (adv_image - original_image).detach()

        if single_image:
            adv_image = adv_image.squeeze(0)
            perturbation = perturbation.squeeze(0)

        return adv_image, perturbation