"""Quantify how smoothly the style space maps onto output images.

Turns the qualitative "t-SNE colours form smooth gradients" claim into citeable
numbers. Samples style codes, renders each on a fixed content image, and measures:

  1. Moran's I of output descriptors over the style space (kNN weights) - spatial
     autocorrelation; ~1 means neighbouring styles give near-identical outputs,
     ~0 means no structure. Computed in the native style space and over the t-SNE
     embedding (the latter validates the figure the reviewer asked for).
  2. Global rank correlation (Spearman) between pairwise style-space distance and
     pairwise output-image distance - does moving in style space move the output.
  3. kNN coherence - how much closer (in output space) a style's style-space
     neighbours are than random styles.

Each Moran's I comes with a permutation p-value. Run from ``implementation/``::

    uv run python src/style_smoothness.py
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from one_to_many_gan.core.generate import GeneratorHandler
from one_to_many_gan.data.config import load_config
from one_to_many_gan.data.datasets import ShoeDataset, dataset_transform


@torch.no_grad()
def sample_and_render(
    handler: GeneratorHandler, s_dim: int, content: torch.Tensor, n: int, batch: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n styles and render each on the fixed content image.

    Returns style directions ``w_hat`` [n, s_dim], per-image descriptors
    [coverage, contrast] [n, 2], and downsampled output vectors [n, 64*32].
    """
    n_blocks = handler.generator.n_style_blocks
    w_hat, descriptors, out_vecs = [], [], []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        z = torch.randn(b, s_dim, device=handler.device)
        w = handler.mapping_network(z)
        s = w[None].expand(n_blocks, -1, -1)
        content_b = content.to(handler.device).expand(b, -1, -1, -1)
        out01 = (handler.generate(content_b, style=s, normalised=True).clamp(-1, 1) + 1) / 2

        small = F.interpolate(out01, size=(64, 32), mode="bilinear", align_corners=False)
        w_np = w.cpu().numpy()
        w_hat.append(w_np / np.clip(np.linalg.norm(w_np, axis=1), 1e-8, None)[:, None])
        descriptors.append(
            torch.stack([out01.mean(dim=(1, 2, 3)), out01.std(dim=(1, 2, 3))], dim=1).cpu().numpy()
        )
        out_vecs.append(small.reshape(b, -1).cpu().numpy())
    return np.concatenate(w_hat), np.concatenate(descriptors), np.concatenate(out_vecs)


def morans_i(values: np.ndarray, coords: np.ndarray, k: int, seed: int, n_perm: int = 999):
    """Moran's I of ``values`` using k-NN weights over ``coords``, with a permutation p."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)[:, 1:]  # drop self

    def statistic(v: np.ndarray) -> float:
        z = v - v.mean()
        return float((z * z[idx].sum(axis=1)).sum() / (k * (z**2).sum()))

    observed = statistic(values)
    rng = np.random.default_rng(seed)
    null = np.array([statistic(rng.permutation(values)) for _ in range(n_perm)])
    p = (1 + (null >= observed).sum()) / (1 + n_perm)
    return observed, p


def knn_coherence(w_hat: np.ndarray, out_vecs: np.ndarray, k: int) -> float:
    """Mean output distance to style-space neighbours / mean output distance to random."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(w_hat)
    idx = nn.kneighbors(w_hat, return_distance=False)[:, 1:]
    neighbour_d = np.mean(
        [np.linalg.norm(out_vecs[i] - out_vecs[idx[i]], axis=1).mean() for i in range(len(w_hat))]
    )
    random_d = pdist(out_vecs).mean()
    return float(neighbour_d / random_d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--n-samples", default=1000, type=int)
    parser.add_argument("--k", default=15, type=int, help="neighbours for Moran's I / coherence")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    handler = GeneratorHandler(config, device)
    s_dim = config["architecture"]["s_dim"]

    transform = dataset_transform(
        config["data"]["image_size"], *config["data"]["shoeprint_norm"], random_image_flip=False
    )
    shoeprints = ShoeDataset(
        config["data"]["shoeprint_data_dir"],
        mode="train",
        transform=transform,
        channels=config["data"]["image_channels"],
    )
    content = shoeprints.random_sample(1)

    print(f"Sampling and rendering {args.n_samples} styles...")
    w_hat, desc, out_vecs = sample_and_render(handler, s_dim, content, args.n_samples)
    coverage, contrast = desc[:, 0], desc[:, 1]
    emb = TSNE(n_components=2, init="pca", random_state=args.seed).fit_transform(w_hat)

    print("Computing metrics...\n")
    # 1. Moran's I in native style space and over the t-SNE embedding.
    results = {}
    for name, coords in [("style-space", w_hat), ("t-SNE", emb)]:
        for label, values in [("coverage", coverage), ("contrast", contrast)]:
            i, p = morans_i(values, coords, args.k, args.seed)
            results[f"{name}/{label}"] = (i, p)

    # 2. Global distance correlation (style-direction vs output image).
    rho, _ = spearmanr(pdist(w_hat), pdist(out_vecs))

    # 3. kNN coherence.
    ratio = knn_coherence(w_hat, out_vecs, args.k)

    print(f"n = {args.n_samples} styles, k = {args.k} neighbours\n")
    print("Moran's I (spatial autocorrelation of output over the map; 1 = perfectly smooth):")
    for key, (i, p) in results.items():
        print(f"  {key:22s}  I = {i:.3f}   (permutation p = {p:.3g})")
    print(f"\nGlobal Spearman rho(style-distance, output-distance) = {rho:.3f}")
    print(f"kNN output distance / random output distance          = {ratio:.3f}")
    print(f"  -> style-space neighbours are {1 / ratio:.1f}x closer in output space than random")


if __name__ == "__main__":
    main()
