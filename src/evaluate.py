"""
evaluate.py
-----------
Evaluation harness for adversarial attacks (FGSM / PGD / DeepFool) on the
face-verification setup in this repo.

What it does
------------
1. Builds a face-VERIFICATION evaluation set from LFW-deepfunneled:
   - "genuine" pairs: two different photos of the SAME identity
   - "impostor" pairs: one photo each from two DIFFERENT identities
2. Extracts embeddings for every image using the same ResNet50 backbone
   the attacks use (fc layer replaced with Identity so we get a 2048-d
   feature vector instead of 1000-way ImageNet class logits).
3. Picks a verification threshold on cosine distance and reports
   clean accuracy / precision / recall / F1 for the "same identity?"
   decision.
4. Re-runs verification on the IMPOSTOR pairs after perturbing the
   source image with the chosen attack (FGSM, PGD, or DeepFool) in
   impersonation mode -- i.e. attacker.attack(source, target=target) --
   for EVERY epsilon in the sweep, and reports per-epsilon:
     - attack success rate (ASR): fraction of impostor pairs the attack
       flips into "verified as same person"
     - accuracy under attack
     - mean AND median embedding distance before/after the attack
       (mean/median similarity drop)
     - average perturbation L-infinity norm (sanity check vs. epsilon)
5. Saves per-epsilon results to results/metrics_<attack>.csv and .json
   so run_visualize.py (or the notebook) can plot them without needing
   to recompute anything.

Usage
-----
    # single epsilon (backwards compatible)
    python evaluate.py --attack fgsm --epsilons 0.03 --n-pairs 200

    # full epsilon sweep, as called out in the progress report
    python evaluate.py --attack fgsm --epsilons 0.01,0.03,0.05,0.1 --n-pairs 200

    # compare attacks (run once per attack, same epsilons, then diff the CSVs)
    python evaluate.py --attack pgd --epsilons 0.01,0.03,0.05,0.1 --n-pairs 200
    python evaluate.py --attack deepfool --epsilons 0.01,0.03,0.05,0.1 --n-pairs 200

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
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import load_pretrained_resnet50
from attack import FGSMAttack, PGDAttack, DeepFoolAttack
from utils import load_image

ATTACKS = {"fgsm": FGSMAttack, "pgd": PGDAttack, "deepfool": DeepFoolAttack}


def build_embedding_model(device):
    """ResNet50 with the classification head removed -> 2048-d embeddings.

    model.py returns a NormalizedModel wrapper (backbone + internal
    ImageNet normalization), so the fc layer lives at model.backbone.fc.
    """
    model = load_pretrained_resnet50(pretrained=True, device=device)
    model.backbone.fc = nn.Identity()
    model.eval()
    return model


def build_attacker(name, model, epsilon, device):
    """Construct an attacker for `name` at the given epsilon budget.

    All three attacks expose the same .attack(image, target=...) API, so
    the rest of the evaluation loop doesn't need to know which one is
    active.
    """
    if name == "fgsm":
        return FGSMAttack(model, epsilon=epsilon, device=device)
    if name == "pgd":
        return PGDAttack(model, epsilon=epsilon, steps=10, device=device)
    if name == "deepfool":
        # DeepFool is a minimal-perturbation attack: it stops as soon as
        # it crosses `threshold`, but max_epsilon caps it so results stay
        # comparable to FGSM/PGD at the same point in the sweep.
        return DeepFoolAttack(model, threshold=0.5, max_iter=50,
                               max_epsilon=epsilon, device=device)
    raise ValueError(f"Unknown attack '{name}', choose from {list(ATTACKS)}")


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


def evaluate_one_epsilon(attack_name, epsilon, embed_model, threshold,
                          impostor_pairs, impostor_embeds, device,
                          progress_every=10):
    """Run the impersonation attack at a single epsilon and return metrics."""
    import time

    attacker = build_attacker(attack_name, embed_model, epsilon, device)

    attack_success = 0
    dists_before, dists_after, linf_norms = [], [], []

    n_total = len(impostor_embeds)
    start = time.time()

    for i, (a_img, b_img, b_emb) in enumerate(impostor_embeds, start=1):
        adv_img, perturbation = attacker.attack(a_img.squeeze(0), target=b_img.squeeze(0))
        adv_emb = embed(embed_model, adv_img.unsqueeze(0), device)

        if i % progress_every == 0 or i == n_total:
            elapsed = time.time() - start
            per_pair = elapsed / i
            eta = per_pair * (n_total - i)
            print(f"    [{attack_name} eps={epsilon}] pair {i}/{n_total}  "
                  f"({per_pair:.2f}s/pair, ~{eta:.0f}s remaining)", flush=True)

        d_before = cosine_distance(embed(embed_model, a_img, device), b_emb)
        d_after = cosine_distance(adv_emb, b_emb)

        dists_before.append(d_before)
        dists_after.append(d_after)
        linf_norms.append(perturbation.abs().max().item())

        if d_after < threshold:  # verifier now (wrongly) says "same identity"
            attack_success += 1

    n = len(impostor_pairs)
    similarity_drop = np.array(dists_after) - np.array(dists_before)  # distance INCREASE = similarity DROP

    return {
        "attack": attack_name,
        "epsilon": epsilon,
        "n_pairs": n,
        "attack_success_rate": attack_success / n,
        "accuracy_under_attack": 1.0 - attack_success / n,
        "mean_distance_before": float(np.mean(dists_before)),
        "median_distance_before": float(np.median(dists_before)),
        "mean_distance_after": float(np.mean(dists_after)),
        "median_distance_after": float(np.median(dists_after)),
        "mean_similarity_drop": float(np.mean(similarity_drop)),
        "median_similarity_drop": float(np.median(similarity_drop)),
        "mean_linf_perturbation": float(np.mean(linf_norms)),
        # raw per-pair values -- kept out of the CSV (summary table) but
        # written to the JSON so run_visualize.py can plot true
        # distributions rather than just the aggregate mean/median.
        "_raw_distances_before": [float(x) for x in dists_before],
        "_raw_distances_after": [float(x) for x in dists_after],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfw-root", default="../data/lfw-deepfunneled/lfw-deepfunneled")
    parser.add_argument("--n-pairs", type=int, default=200,
                         help="number of genuine pairs AND number of impostor pairs")
    parser.add_argument("--attack", choices=list(ATTACKS), default="fgsm")
    parser.add_argument("--epsilons", type=str, default="0.01,0.03,0.05,0.1",
                         help="comma-separated list of epsilon (L-inf budget) values")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="../results")
    args = parser.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(",")]

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

    # ---------- 1. Clean verification metrics (epsilon-independent) ----------
    genuine_dists, impostor_dists = [], []

    for a_path, b_path, _, _ in genuine_pairs:
        a = embed(embed_model, load_image(a_path), args.device)
        b = embed(embed_model, load_image(b_path), args.device)
        genuine_dists.append(cosine_distance(a, b))

    impostor_embeds = []  # cache so we don't reload/re-embed per epsilon
    for a_path, b_path, _, _ in impostor_pairs:
        a_img = load_image(a_path)
        b_img = load_image(b_path)
        b_emb = embed(embed_model, b_img, args.device)
        impostor_dists.append(
            cosine_distance(embed(embed_model, a_img, args.device), b_emb)
        )
        impostor_embeds.append((a_img, b_img, b_emb))

    threshold = best_threshold(genuine_dists, impostor_dists)

    labels = [1] * len(genuine_dists) + [0] * len(impostor_dists)
    preds = [int(d < threshold) for d in genuine_dists + impostor_dists]
    clean_acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    print("\n============= CLEAN VERIFICATION =============")
    print(f"Verification threshold (cosine dist): {threshold:.4f}")
    print(f"Clean accuracy:  {clean_acc*100:.2f}%")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-score:        {f1:.4f}")
    print("================================================\n")

    # ---------- 2. Epsilon sweep for the chosen attack ----------
    rows = []
    print(f"Running {args.attack.upper()} epsilon sweep: {epsilons}")
    for eps in epsilons:
        metrics = evaluate_one_epsilon(
            args.attack, eps, embed_model, threshold,
            impostor_pairs, impostor_embeds, args.device,
        )
        rows.append(metrics)
        print(f"  eps={eps:<6} ASR={metrics['attack_success_rate']*100:6.2f}%  "
              f"mean_sim_drop={metrics['mean_similarity_drop']:.4f}  "
              f"median_sim_drop={metrics['median_similarity_drop']:.4f}  "
              f"mean_linf={metrics['mean_linf_perturbation']:.4f}")

    # ---------- 3. Save results ----------
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = [{k: v for k, v in r.items() if not k.startswith("_raw_")} for r in rows]
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output_dir, f"metrics_{args.attack}.csv")
    json_path = os.path.join(args.output_dir, f"metrics_{args.attack}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({
            "attack": args.attack,
            "threshold": threshold,
            "clean_accuracy": clean_acc,
            "clean_precision": precision,
            "clean_recall": recall,
            "clean_f1": f1,
            "n_pairs": args.n_pairs,
            "epsilons": rows,
        }, f, indent=2)

    print(f"\nSaved per-epsilon metrics to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()