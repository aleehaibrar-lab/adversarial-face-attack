"""
run_visualize.py
-----------------
Turns the metrics produced by evaluate.py into plots, covering the
report's "Qualitative inspection of perturbation visibility versus
attack strength" and "Visualize cosine similarity / L2 distance
distributions across epsilon values" objectives.

Three things get plotted:

1. Attack Success Rate vs. epsilon (one line per attack that has a
   metrics_<attack>.json file in --results-dir).
2. Cosine-distance distributions (before vs. after the attack) as
   histograms, one subplot per epsilon, for a chosen attack.
3. A qualitative grid: the same source/target pair attacked at every
   epsilon in the sweep, showing the adversarial image and the
   (rescaled-for-visibility) perturbation side by side, so you can see
   how visible the perturbation gets as epsilon grows.

Usage
-----
    # after running evaluate.py for one or more attacks:
    python run_visualize.py --results-dir ../results --attacks fgsm,pgd,deepfool

    # qualitative grid needs a live model + sample images (not just the
    # saved metrics), so it's a separate optional step:
    python run_visualize.py --qualitative --attack fgsm \
        --epsilons 0.01,0.03,0.05,0.1 \
        --source ../data/sample_images/test_image.jpg \
        --target ../data/sample_images/target_image.jpg
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; figures are saved to disk, not shown
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(results_dir, attack):
    path = os.path.join(results_dir, f"metrics_{attack}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_asr_vs_epsilon(results_dir, attacks, out_path):
    """One ASR-vs-epsilon line per attack, so FGSM/PGD/DeepFool are
    directly comparable at the same perturbation budget."""
    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False

    for attack in attacks:
        data = load_metrics(results_dir, attack)
        if data is None:
            print(f"  [skip] no metrics_{attack}.json in {results_dir}")
            continue
        any_data = True
        eps = [row["epsilon"] for row in data["epsilons"]]
        asr = [row["attack_success_rate"] * 100 for row in data["epsilons"]]
        ax.plot(eps, asr, marker="o", label=attack.upper())

    if not any_data:
        print("No metrics files found -- run evaluate.py first.")
        plt.close(fig)
        return

    ax.set_xlabel("Epsilon (L-infinity perturbation budget)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Attack Success Rate vs. Epsilon")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_distance_distributions(results_dir, attack, out_path):
    """Histogram of cosine distance before vs. after the attack, one
    subplot per epsilon."""
    data = load_metrics(results_dir, attack)
    if data is None:
        print(f"No metrics_{attack}.json in {results_dir} -- run evaluate.py first.")
        return

    rows = data["epsilons"]
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        before = row.get("_raw_distances_before")
        after = row.get("_raw_distances_after")
        if before is None or after is None:
            ax.set_title(f"eps={row['epsilon']}\n(no raw data saved)")
            continue
        bins = np.linspace(0, max(max(before), max(after)) * 1.05, 25)
        ax.hist(before, bins=bins, alpha=0.6, label="before attack")
        ax.hist(after, bins=bins, alpha=0.6, label="after attack")
        ax.axvline(data["threshold"], color="black", linestyle="--",
                   linewidth=1, label="verification threshold")
        ax.set_title(f"{attack.upper()}  eps={row['epsilon']}\n"
                     f"ASR={row['attack_success_rate']*100:.1f}%")
        ax.set_xlabel("Cosine distance to target")

    axes[0].set_ylabel("Number of impostor pairs")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{attack.upper()}: embedding distance to target, before vs. after attack")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def qualitative_grid(attack_name, epsilons, source_path, target_path, out_path, device="cpu"):
    """Run the attack at every epsilon on one source/target pair and show
    the resulting adversarial image + rescaled perturbation side by side,
    so perturbation visibility vs. attack strength can be inspected by eye.
    """
    # local imports so plotting-only usage doesn't require torch/torchvision
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    import torch
    import torch.nn as nn
    from model import load_pretrained_resnet50
    from evaluate import build_attacker
    from utils import load_image

    model = load_pretrained_resnet50(pretrained=True, device=device)
    model.backbone.fc = nn.Identity()
    model.eval()

    source = load_image(source_path).squeeze(0)
    target = load_image(target_path).squeeze(0)

    fig, axes = plt.subplots(2, len(epsilons) + 1, figsize=(3.2 * (len(epsilons) + 1), 6.4))

    def to_numpy_img(t):
        return t.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)

    axes[0, 0].imshow(to_numpy_img(source))
    axes[0, 0].set_title("source (original)")
    axes[1, 0].imshow(to_numpy_img(target))
    axes[1, 0].set_title("target (impersonation goal)")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col, eps in enumerate(epsilons, start=1):
        attacker = build_attacker(attack_name, model, eps, device)
        adv_img, perturbation = attacker.attack(source, target=target)

        axes[0, col].imshow(to_numpy_img(adv_img))
        axes[0, col].set_title(f"adversarial\nepsilon={eps}")
        axes[0, col].axis("off")

        pert = perturbation.detach().cpu().numpy().transpose(1, 2, 0)
        pert_vis = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
        axes[1, col].imshow(pert_vis)
        axes[1, col].set_title(f"perturbation\n(rescaled for visibility)")
        axes[1, col].axis("off")

    fig.suptitle(f"{attack_name.upper()}: perturbation visibility vs. attack strength")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--attacks", default="fgsm,pgd,deepfool",
                         help="comma-separated attacks to include in the ASR-vs-epsilon plot")
    parser.add_argument("--attack", default="fgsm",
                         help="single attack to use for the distribution / qualitative plots")
    parser.add_argument("--epsilons", default="0.01,0.03,0.05,0.1")
    parser.add_argument("--qualitative", action="store_true",
                         help="also produce the perturbation-visibility grid (needs the model + sample images)")
    parser.add_argument("--source", default="../data/sample_images/test_image.jpg")
    parser.add_argument("--target", default="../data/sample_images/target_image.jpg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="../results/visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    attacks = [a.strip() for a in args.attacks.split(",")]

    plot_asr_vs_epsilon(
        args.results_dir, attacks,
        os.path.join(args.output_dir, "asr_vs_epsilon.png"),
    )
    plot_distance_distributions(
        args.results_dir, args.attack,
        os.path.join(args.output_dir, f"distance_distributions_{args.attack}.png"),
    )

    if args.qualitative:
        epsilons = [float(e) for e in args.epsilons.split(",")]
        qualitative_grid(
            args.attack, epsilons, args.source, args.target,
            os.path.join(args.output_dir, f"perturbation_visibility_{args.attack}.png"),
            device=args.device,
        )


if __name__ == "__main__":
    main()
