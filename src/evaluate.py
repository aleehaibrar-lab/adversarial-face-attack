"""
evaluate.py
-----------
Evaluation harness for the FGSM impersonation attack in this repo.

The rest of the codebase (attack.py, model.py, utils.py, the notebook)
only ever runs the attack on a single source/target image pair and saves
the four result images -- there is no loop over the dataset and no
accuracy/precision/recall computation anywhere. This script adds that.

What it does
------------
1. Builds a face-VERIFICATION evaluation set from LFW-deepfunneled:
   - "genuine" pairs: two different photos of the SAME identity
   - "impostor" pairs: one photo each from two DIFFERENT identities
2. Extracts embeddings for every image using the same ResNet50 backbone
   the attack uses (fc layer replaced with Identity so we get a 2048-d
   feature vector instead of 1000-way ImageNet class logits -- this is
   what FGSMAttack's targeted/impersonation mode actually pushes on).
3. Picks a verification threshold on cosine distance and reports
   clean accuracy / precision / recall / F1 for the "same identity?"
   decision.
4. Re-runs verification on the IMPOSTOR pairs after perturbing the
   source image with FGSMAttack(target=target_image) -- i.e. the
   impersonation attack -- and reports:
     - attack success rate (fraction of impostor pairs the attack
       flips into "verified as same person")
     - accuracy under attack (does the verifier still correctly say
       "different person" despite the attack?)
     - average embedding distance reduction caused by the attack
     - average perturbation L-infinity norm (sanity check vs. epsilon)

Usage
-----
    python evaluate.py --lfw-root ../data/lfw-deepfunneled/lfw-deepfunneled \
                        --n-pairs 200 --epsilon 0.3 --device cpu

Notes
-----
- Needs internet access the first time it runs (torchvision downloads
  ImageNet-pretrained ResNet50 weights).
- "Accuracy" here is defined for the verification task (same/different
  identity), which is the standard way to evaluate face-recognition
  attacks -- ResNet50 on raw ImageNet weights is not a face-recognition
  model, so this is illustrative, not a state-of-the-art benchmark.
  For a real system, swap in a proper face embedding model
  (e.g. FaceNet / ArcFace) via model.py.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import load_pretrained_resnet50
from attack import FGSMAttack
from utils import load_image


def build_embedding_model(device):
    """ResNet50 with the classification head removed -> 2048-d embeddings."""
    model = load_pretrained_resnet50(pretrained=True, device=device)
    model.fc = nn.Identity()
    model.eval()
    return model


def list_identities(lfw_root):
    """Return {identity_name: [image_paths]} for every folder with >=1 image."""
    identities = {}
    for name in sorted(os.listdir(lfw_root)):
        folder = os.path.join(lfw_root, name)
        if not os.path.isdir(folder):
            continue
        imgs = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if imgs:
            identities[name] = imgs
    return identities


def sample_pairs(identities, n_pairs, seed=0):
    """Sample balanced genuine / impostor pairs.

    genuine pair  = (img_a, img_b) from the SAME identity (needs >=2 images)
    impostor pair = (img_a, img_b) from two DIFFERENT identities
    """
    rng = random.Random(seed)

    multi_image_ids = [name for name, imgs in identities.items() if len(imgs) >= 2]
    all_ids = list(identities.keys())

    genuine_pairs = []
    for _ in range(n_pairs):
        name = rng.choice(multi_image_ids)
        a, b = rng.sample(identities[name], 2)
        genuine_pairs.append((a, b, name, name))

    impostor_pairs = []
    for _ in range(n_pairs):
        name_a, name_b = rng.sample(all_ids, 2)
        a = rng.choice(identities[name_a])
        b = rng.choice(identities[name_b])
        impostor_pairs.append((a, b, name_a, name_b))

    return genuine_pairs, impostor_pairs


@torch.no_grad()
def embed(model, image_tensor, device):
    return model(image_tensor.to(device)).squeeze(0)


def cosine_distance(a, b):
    a = a / a.norm()
    b = b / b.norm()
    return 1.0 - torch.dot(a, b).item()


def best_threshold(genuine_dists, impostor_dists):
    """Sweep thresholds, pick the one maximizing accuracy on this set."""
    all_dists = np.array(sorted(genuine_dists + impostor_dists))
    labels = np.array([1] * len(genuine_dists) + [0] * len(impostor_dists))
    dists = np.array(genuine_dists + impostor_dists)

    best_acc, best_t = 0.0, all_dists[0]
    for t in all_dists:
        preds = (dists < t).astype(int)  # 1 = "same identity"
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfw-root", default="../data/lfw-deepfunneled/lfw-deepfunneled")
    parser.add_argument("--n-pairs", type=int, default=200,
                         help="number of genuine pairs AND number of impostor pairs")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading identities from {args.lfw_root} ...")
    identities = list_identities(args.lfw_root)
    print(f"Found {len(identities)} identities, "
          f"{sum(len(v) for v in identities.values())} images total.")

    genuine_pairs, impostor_pairs = sample_pairs(identities, args.n_pairs, args.seed)
    print(f"Sampled {len(genuine_pairs)} genuine and {len(impostor_pairs)} impostor pairs.")

    print("Loading embedding model (ResNet50, fc removed) ...")
    embed_model = build_embedding_model(args.device)
    attacker = FGSMAttack(embed_model, epsilon=args.epsilon, device=args.device)

    # ---------- 1. Clean verification metrics ----------
    genuine_dists, impostor_dists = [], []

    for a_path, b_path, _, _ in genuine_pairs:
        a = embed(embed_model, load_image(a_path), args.device)
        b = embed(embed_model, load_image(b_path), args.device)
        genuine_dists.append(cosine_distance(a, b))

    impostor_embeds = []  # cache so we don't recompute for the attack phase
    for a_path, b_path, _, _ in impostor_pairs:
        a_img = load_image(a_path)
        b_img = load_image(b_path)
        a = embed(embed_model, a_img, args.device)
        b = embed(embed_model, b_img, args.device)
        impostor_dists.append(cosine_distance(a, b))
        impostor_embeds.append((a_img, b_img, b))  # keep target image + its embedding

    threshold = best_threshold(genuine_dists, impostor_dists)

    labels = [1] * len(genuine_dists) + [0] * len(impostor_dists)
    preds = [int(d < threshold) for d in genuine_dists + impostor_dists]
    clean_acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    # ---------- 2. Attack evaluation (impostor pairs only) ----------
    attack_success = 0
    post_attack_dists = []
    linf_norms = []

    for (a_path, b_path, _, _), (a_img, b_img, b_emb) in zip(impostor_pairs, impostor_embeds):
        adv_img, perturbation = attacker.attack(a_img.squeeze(0), target=b_img.squeeze(0))
        adv_emb = embed(embed_model, adv_img.unsqueeze(0), args.device)

        d_after = cosine_distance(adv_emb, b_emb)
        post_attack_dists.append(d_after)
        linf_norms.append(perturbation.abs().max().item())

        if d_after < threshold:  # verifier now (wrongly) says "same identity"
            attack_success += 1

    n_impostor = len(impostor_pairs)
    attack_success_rate = attack_success / n_impostor
    accuracy_under_attack = 1.0 - attack_success_rate  # correct "different" calls remaining
    avg_dist_before = float(np.mean(impostor_dists))
    avg_dist_after = float(np.mean(post_attack_dists))
    avg_linf = float(np.mean(linf_norms))

    # ---------- 3. Report ----------
    print("\n================ RESULTS ================")
    print(f"Verification threshold (cosine dist): {threshold:.4f}")
    print(f"Clean accuracy:        {clean_acc*100:.2f}%")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1-score:              {f1:.4f}")
    print("-------------------------------------------")
    print(f"Attack success rate:    {attack_success_rate*100:.2f}%")
    print(f"Accuracy under attack:  {accuracy_under_attack*100:.2f}%")
    print(f"Avg. distance before attack: {avg_dist_before:.4f}")
    print(f"Avg. distance after attack:  {avg_dist_after:.4f}")
    print(f"Avg. perturbation L-inf norm: {avg_linf:.4f} (epsilon={args.epsilon})")
    print("=============================================")


if __name__ == "__main__":
    main()
