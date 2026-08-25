"""Analyse the disentangled style space S of the trained mapping network.

The mapping network maps ``w ~ N(0, I)`` (in R^s_dim, L2-normalised onto the unit
sphere) to a non-negative style vector ``s`` (the final layer is a ReLU). The domain
variable ``theta`` in [0, 1] then *scales* that vector before it is fed to the
generator. So the style has a polar structure:

    * direction  s_hat = s / ||s||   -> "style type"
    * magnitude  ||s|| * theta        -> "intensity" (theta is an explicit knob)

Because theta is an external knob, clustering is analysed on the *direction* s_hat,
otherwise the embedding would just separate large-theta from small-theta samples.

Notation follows the thesis: ``w`` is the Gaussian latent, ``s`` the style vector.

Outputs (written to ``--out-dir``, all without embedded titles so they can be
dropped straight into the thesis):

    * ``s_raw.npy``            - sampled style matrix, for offline re-analysis.
    * ``s_pca_scree.png``      - PCA explained variance (intrinsic dimensionality).
    * ``s_t_sne_norm.png``     - t-SNE of s_hat coloured by ||s||.
    * ``s_t_sne_coverage.png`` - t-SNE of s_hat coloured by output coverage.
    * ``s_t_sne_contrast.png`` - t-SNE of s_hat coloured by output contrast.
    * ``s_pca_scatter.png``    - 2-D PCA of s_hat (linear, direction-preserving).
    * ``walk_pca.png``         - GANSpace walk: content + theta fixed, move along the
                                 top PCA directions of S and render the generator.
    * ``walk_sefa.png``        - SeFa walk: same, along closed-form directions from the
                                 modulation (``to_style``) weights.

t-SNE reveals *clusters* but distorts global geometry / directions; the PCA and SeFa
walks are the parts that actually show how *directionality* in style space maps to
image change, so both are produced.

Run from the ``implementation`` directory::

    uv run python src/analyse_style_space.py
"""

import argparse
import random
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from one_to_many_gan.core.generate import GeneratorHandler
from one_to_many_gan.data.config import Config, load_config
from one_to_many_gan.data.datasets import ShoeDataset, dataset_transform
from one_to_many_gan.model.builder import StyleExtractor
from one_to_many_gan.model.layers import Conv2dWeightModulate


@torch.no_grad()
def sample_s(handler: GeneratorHandler, s_dim: int, n: int, batch: int = 512) -> np.ndarray:
    """Sample n Gaussian latents and map them to the style space S."""
    ss = []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        w = torch.randn(b, s_dim, device=handler.device)
        ss.append(handler.mapping_network(w).cpu())
    return torch.cat(ss).numpy()


@torch.no_grad()
def render(
    handler: GeneratorHandler, shoeprint: torch.Tensor, s: torch.Tensor, theta: float = 1.0
) -> torch.Tensor:
    """Render one generator output per style vector on a fixed content image.

    ``shoeprint`` is a single normalised content image ``[1, C, H, W]``; ``s`` is
    ``[b, s_dim]``; ``theta`` is the domain variable that scales the style.
    Returns images in ``[-1, 1]``.
    """
    n_blocks = handler.generator.n_style_blocks
    style = (s.to(handler.device) * theta)[None].expand(n_blocks, -1, -1)
    content = shoeprint.to(handler.device).expand(s.shape[0], -1, -1, -1)
    return handler.generate(content, style=style, normalised=True)


def load_style_extractor(config: Config, device: torch.device) -> StyleExtractor:
    """Load the trained StyleExtractor (image -> style vector) from the checkpoint."""
    extractor = (
        StyleExtractor(
            input_nc=config["data"]["image_channels"], s_dim=config["architecture"]["s_dim"]
        )
        .to(device)
        .eval()
    )
    checkpoint = torch.load(config["inference"]["checkpoint"], map_location=device)
    state = {
        k.removeprefix("_orig_mod."): v
        for k, v in checkpoint["style_extractor_state_dict"].items()
    }
    extractor.load_state_dict(state)
    for param in extractor.parameters():
        param.requires_grad = False
    return extractor


@torch.no_grad()
def output_descriptors(
    handler: GeneratorHandler, s: np.ndarray, shoeprint: torch.Tensor, batch: int = 16
) -> dict[str, np.ndarray]:
    """Per-style output descriptors, grounding the style clusters in image space.

    ``coverage`` is mean pixel intensity (how much ink / degradation), ``contrast`` is
    the per-image standard deviation. Both are cheap, interpretable shoemark
    attributes to colour the embedding by.
    """
    coverage, contrast = [], []
    for start in range(0, len(s), batch):
        sb = torch.from_numpy(s[start : start + batch])
        out01 = (render(handler, shoeprint, sb).clamp(-1, 1) + 1) / 2
        coverage.append(out01.mean(dim=(1, 2, 3)).cpu())
        contrast.append(out01.std(dim=(1, 2, 3)).cpu())
    return {
        "coverage": torch.cat(coverage).numpy(),
        "contrast": torch.cat(contrast).numpy(),
    }


def sefa_directions(handler: GeneratorHandler, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form SeFa directions from the modulation (``to_style``) weights.

    Every ``Conv2dWeightModulate`` has a linear ``to_style: s_dim -> in_features``.
    Stacking those weight matrices and eigendecomposing ``A^T A`` yields the style-space
    directions that most strongly drive the modulation, with no sampling required.
    Returns the top-k directions ``[k, s_dim]`` and their eigenvalues.
    """
    mats = [
        m.to_style.weight().detach().cpu().numpy()  # [in_features, s_dim] (equalised)
        for m in handler.generator.modules()
        if isinstance(m, Conv2dWeightModulate)
    ]
    a = np.concatenate(mats, axis=0)
    eigvals, eigvecs = np.linalg.eigh(a.T @ a)  # ascending
    order = eigvals.argsort()[::-1]
    return eigvecs[:, order[:k]].T, eigvals[order[:k]]


def save_walk(
    handler: GeneratorHandler,
    shoeprint: torch.Tensor,
    base: np.ndarray,
    directions: np.ndarray,
    scales: np.ndarray,
    steps: np.ndarray,
    path: Path,
) -> None:
    """Render a grid: one row per direction, walking base +/- scale*step along it.

    Content image and theta are held fixed, so every change in the grid is
    attributable to movement in the style space alone.
    """
    rows = []
    for direction, scale in zip(directions, scales, strict=True):
        edits = base[None] + steps[:, None] * scale * direction[None]
        rows.append(render(handler, shoeprint, torch.from_numpy(edits.astype(np.float32))))
    grid = torchvision.utils.make_grid(
        (torch.cat(rows).clamp(-1, 1) + 1) / 2, nrow=len(steps)
    )
    plt.figure(figsize=(len(steps) * 1.6, len(directions) * 1.6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def maximin_styles(
    handler: GeneratorHandler,
    content: torch.Tensor,
    s: np.ndarray,
    n_rows: int,
    n_candidates: int = 600,
    size: tuple[int, int] = (128, 64),
    batch: int = 16,
) -> np.ndarray:
    """Pick ``n_rows`` maximally different-looking styles by farthest-point selection.

    Greedy maximin over pairwise LPIPS of the candidates' renders (theta = 1, fixed
    content, downsampled): start from a random style, then repeatedly add the style
    whose minimum LPIPS distance to the already-chosen set is largest. Being greedy
    over render distance rather than style-space distance selects styles that are as
    *visually* distinct as the model can make them, which is what the theta grid
    should showcase.
    """
    cand = s[np.random.choice(len(s), min(n_candidates, len(s)), replace=False)]

    imgs = []
    for start in range(0, len(cand), batch):
        sb = torch.from_numpy(cand[start : start + batch].astype(np.float32))
        out = render(handler, content, sb).clamp(-1, 1)
        imgs.append(F.interpolate(out, size=size, mode="bilinear", align_corners=False).cpu())
    imgs = torch.cat(imgs).repeat(1, 3, 1, 1)  # LPIPS wants 3 channels

    device = handler.device
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def dist_to_all(i: int) -> np.ndarray:
        a = imgs[i][None].to(device)
        out = []
        for start in range(0, len(imgs), 256):
            b = imgs[start : start + 256].to(device)
            out.append(lp(a.expand(len(b), -1, -1, -1), b).flatten().cpu().numpy())
        return np.concatenate(out)

    chosen = [int(np.random.randint(len(cand)))]
    min_d = dist_to_all(chosen[0])
    while len(chosen) < n_rows:
        nxt = int(min_d.argmax())  # self-distance is 0, so chosen can't repeat
        chosen.append(nxt)
        min_d = np.minimum(min_d, dist_to_all(nxt))
    return cand[chosen]


@torch.no_grad()
def theta_interaction(
    handler: GeneratorHandler,
    content: torch.Tensor,
    reps: np.ndarray,
    thetas: np.ndarray,
    out_dir: Path,
) -> None:
    """Sweep the domain variable theta across maximally different styles.

    ``reps`` are the styles selected by :func:`maximin_styles` (rows); theta sweeps
    across the columns. Both render the grid and plot output coverage vs theta: if
    the coverage curves for different styles merely overlay, the styles only differ
    in *where* they place structure, not in overall intensity; offset or differently
    shaped curves mean the style also modulates how output builds up with theta.
    """
    rows, coverage = [], []
    for sv in reps:
        sv_t = torch.from_numpy(sv[None].astype(np.float32))
        imgs = [render(handler, content, sv_t, theta=float(t)) for t in thetas]
        rows.append(torch.cat(imgs))
        out01 = [(im.clamp(-1, 1) + 1) / 2 for im in imgs]
        coverage.append([float(im.mean()) for im in out01])

    grid = torchvision.utils.make_grid((torch.cat(rows).clamp(-1, 1) + 1) / 2, nrow=len(thetas))
    pad = 2  # make_grid default padding
    _, grid_h, grid_w = grid.shape
    cell_w = (grid_w - pad) / len(thetas)
    cell_h = (grid_h - pad) / len(reps)
    plt.figure(figsize=(len(thetas) * 1.6, len(reps) * 1.6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.xticks([pad / 2 + (i + 0.5) * cell_w for i in range(len(thetas))],
               [f"{t:g}" for t in thetas])
    plt.yticks([pad / 2 + (i + 0.5) * cell_h for i in range(len(reps))],
               [str(i + 1) for i in range(len(reps))])
    plt.tick_params(length=0)
    plt.xlabel(r"$\theta$")
    plt.ylabel("style")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "theta_interaction.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 5))
    for i, cov in enumerate(coverage):
        plt.plot(thetas, cov, "o-", label=f"style {i + 1}")
    plt.xlabel(r"$\theta$")
    plt.ylabel("mean output intensity")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "theta_coverage.png", dpi=150)
    plt.close()


@torch.no_grad()
def real_overlay(
    config: Config,
    device: torch.device,
    s_hat: np.ndarray,
    pca_dir: PCA,
    out_dir: Path,
    n_real: int,
    seed: int,
) -> None:
    """Overlay real shoemark styles (via StyleExtractor) on the sampled style map.

    Tests whether the learned style space actually covers real-data variation: real
    styles landing inside the sampled cloud means good coverage; real styles outside it
    means the generator can't reach part of the real distribution. Real styles are
    projected onto the same PCA basis (linear) and also embedded in a joint t-SNE.
    """
    extractor = load_style_extractor(config, device)
    transform = dataset_transform(
        config["data"]["image_size"], *config["data"]["shoemark_norm"], random_image_flip=False
    )
    # Held-out marks: the style losses regularise extracted styles of *training*
    # marks towards the cone, so only unseen marks test coverage meaningfully.
    shoemarks = ShoeDataset(
        config["data"]["shoemark_data_dir"],
        mode="val",
        transform=transform,
        channels=config["data"]["image_channels"],
    )
    n_real = min(n_real, len(shoemarks))
    reals = shoemarks.random_sample(n_real)

    extracted = []
    for start in range(0, n_real, 16):
        batch = reals[start : start + 16].to(device)
        extracted.append(extractor(batch).cpu())
    real_s = torch.cat(extracted).numpy()
    real_dir = real_s / np.clip(np.linalg.norm(real_s, axis=1), 1e-8, None)[:, None]

    # Fraction of real styles falling outside the non-negative orthant the generator
    # produces (its final layer is ReLU, so sampled styles are all >= 0).
    outside = float((real_s < 0).any(axis=1).mean())
    print(f"{outside:.0%} of real styles have a negative (off-manifold) component")

    sampled_2d = pca_dir.transform(s_hat)
    real_2d = pca_dir.transform(real_dir)
    plt.figure(figsize=(6, 5))
    plt.scatter(sampled_2d[:, 0], sampled_2d[:, 1], s=6, c="lightgray", label="sampled (generator)")
    plt.scatter(real_2d[:, 0], real_2d[:, 1], s=10, c="crimson", label="real (extracted)", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "real_overlay_pca.png", dpi=150)
    plt.close()

    joint = np.concatenate([s_hat, real_dir], axis=0)
    emb = TSNE(n_components=2, init="pca", random_state=seed).fit_transform(joint)
    plt.figure(figsize=(6, 5))
    plt.scatter(emb[: len(s_hat), 0], emb[: len(s_hat), 1], s=6, c="lightgray", label="sampled")
    plt.scatter(emb[len(s_hat):, 0], emb[len(s_hat):, 1], s=10, c="crimson", label="real", alpha=0.6)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / "real_overlay_tsne.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--out-dir", default="style_analysis", type=Path)
    parser.add_argument("--n-samples", default=4000, type=int)
    parser.add_argument("--n-directions", default=6, type=int, help="directions to walk")
    parser.add_argument("--walk-sigma", default=2.0, type=float, help="+/- range of the walk")
    parser.add_argument("--walk-steps", default=7, type=int)
    parser.add_argument("--perplexity", default=30.0, type=float)
    parser.add_argument("--theta-steps", default=6, type=int, help="theta values in the sweep")
    parser.add_argument("--n-style-rows", default=5, type=int, help="styles in the theta grid")
    parser.add_argument("--n-real", default=500, type=int, help="real shoemarks to overlay")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)  # ShoeDataset.random_sample draws the content image

    config = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    handler = GeneratorHandler(config, device)
    s_dim = config["architecture"]["s_dim"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # A single fixed content image so output changes are attributable to style only.
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

    # 1. Sample the style space S.
    print(f"Sampling {args.n_samples} style vectors...")
    s = sample_s(handler, s_dim, args.n_samples)
    np.save(args.out_dir / "s_raw.npy", s)
    magnitude = np.linalg.norm(s, axis=1)
    s_hat = s / np.clip(magnitude, 1e-8, None)[:, None]  # direction only

    # 2. PCA: intrinsic dimensionality of the style space.
    pca = PCA().fit(s)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, s_dim + 1), np.cumsum(pca.explained_variance_ratio_), "o-")
    plt.xlabel("component")
    plt.ylabel("cumulative explained variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "s_pca_scree.png", dpi=150)
    plt.close()

    pca_dir = PCA(n_components=2).fit(s_hat)
    pca2 = pca_dir.transform(s_hat)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(pca2[:, 0], pca2[:, 1], c=magnitude, s=6, cmap="viridis")
    plt.colorbar(sc, label=r"$||\mathbf{s}||$")
    plt.tight_layout()
    plt.savefig(args.out_dir / "s_pca_scatter.png", dpi=150)
    plt.close()

    # 3. Output descriptors to ground the clustering in image space.
    print("Rendering output descriptors...")
    desc = output_descriptors(handler, s, content)

    # 4. t-SNE of the style direction, coloured by magnitude and output descriptors,
    #    one figure per colouring (they appear as separate subfloats in the thesis).
    print("Running t-SNE...")
    emb = TSNE(
        n_components=2, perplexity=args.perplexity, init="pca", random_state=args.seed
    ).fit_transform(s_hat)
    colourings = {
        "s_t_sne_norm": (r"$||\mathbf{s}||$", magnitude),
        "s_t_sne_coverage": ("mean output intensity", desc["coverage"]),
        "s_t_sne_contrast": ("output contrast", desc["contrast"]),
    }
    for fname, (label, values) in colourings.items():
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(emb[:, 0], emb[:, 1], c=values, s=6, cmap="viridis")
        plt.colorbar(sc, label=label)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(args.out_dir / f"{fname}.png", dpi=150)
        plt.close()

    # 5. Directionality walks. base = mean style; walk along top PCA and SeFa directions.
    base = s.mean(axis=0)
    steps = np.linspace(-args.walk_sigma, args.walk_sigma, args.walk_steps)

    pca_dirs = PCA(n_components=args.n_directions).fit(s)
    save_walk(
        handler, content, base,
        pca_dirs.components_,
        np.sqrt(pca_dirs.explained_variance_),  # scale each walk by that PC's spread
        steps,
        args.out_dir / "walk_pca.png",
    )

    sefa_dirs, _ = sefa_directions(handler, args.n_directions)
    save_walk(
        handler, content, base,
        sefa_dirs,
        np.full(args.n_directions, magnitude.mean()),  # scale by typical style magnitude
        steps,
        args.out_dir / "walk_sefa.png",
    )

    # 6. theta sweep over maximally different-looking styles.
    print("Selecting maximin-LPIPS styles and rendering theta interaction...")
    reps = maximin_styles(handler, content, s, args.n_style_rows)
    theta_interaction(
        handler,
        content,
        reps,
        np.linspace(0.0, 1.0, args.theta_steps),
        args.out_dir,
    )

    # 7. Overlay real shoemark styles on the sampled style map.
    print("Overlaying real shoemark styles...")
    real_overlay(config, device, s_hat, pca_dir, args.out_dir, args.n_real, args.seed)

    print(f"Done. Wrote analysis to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
