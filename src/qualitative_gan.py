"""Qualitative figures for the GAN chapter, with maximally distinct styles.

Replaces the style-diversity figures previously produced by generate_diagrams.py.
Styles are chosen by farthest-point (maximin) selection over LPIPS distance
between their rendered outputs, so the marks displayed span the range the model
can produce rather than an arbitrary draw that can land on near-duplicates. This
mirrors the selection used for the UNSB chapter's varied-styles figure, so the
two chapters' figures are directly comparable.

The selected styles are ordered by ink coverage, densest first, so a figure row
reads as increasing degradation.

Note on normalisation: the dataset transform already normalises, so generate()
is called with normalised=True, matching the metric and style-space scripts.
generate_diagrams.py omits this and so normalises twice, but the result is
bit-identical: the encoder's first block is a convolution followed by an
InstanceNorm2d, which is invariant to an affine rescaling of its input.

    uv run python src/qualitative_gan.py [--n-styles 4] [--out-dir ...]
"""

import argparse
import random
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from one_to_many_gan.core.generate import GeneratorHandler
from one_to_many_gan.data.config import load_config
from one_to_many_gan.data.datasets import ShoeDataset, dataset_transform

N_CAND = 256          # candidate styles the selection draws from
LPIPS_SIZE = (128, 64)
BATCH = 16


@torch.no_grad()
def render(handler, styles, content, size=None):
    """One output per style vector on a fixed content image. Styles are [b, s_dim]
    at full magnitude (theta = 1); content is already normalised."""
    n_blocks = handler.generator.n_style_blocks
    outs = []
    for start in range(0, len(styles), BATCH):
        sb = styles[start:start + BATCH].to(handler.device)
        s = sb[None].expand(n_blocks, -1, -1)
        cb = content.to(handler.device).expand(sb.shape[0], -1, -1, -1)
        out = handler.generate(cb, style=s, normalised=True).clamp(-1, 1)
        if size is not None:
            out = F.interpolate(out, size, mode="bilinear", align_corners=False)
        outs.append(out.cpu())
    return torch.cat(outs)


@torch.no_grad()
def maximin_styles(handler, content, styles, n, device, lp):
    """Greedy farthest-point selection over LPIPS distance between renders."""
    small = render(handler, styles, content, LPIPS_SIZE).repeat(1, 3, 1, 1)

    def dist_to(i):
        out = []
        for s in range(0, len(small), 128):
            b = small[s:s + 128].to(device)
            out.append(lp(small[i][None].to(device).expand(len(b), -1, -1, -1), b)
                       .flatten().cpu().numpy())
        return np.concatenate(out)

    chosen = [0]
    mind = dist_to(0)
    while len(chosen) < n:
        nxt = int(mind.argmax())
        chosen.append(nxt)
        mind = np.minimum(mind, dist_to(nxt))
    return styles[chosen]


def save(x, path):
    torchvision.utils.save_image(x, path, normalize=True, value_range=(-1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--n-styles", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=Path("style_analysis") / "qualitative")
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    handler = GeneratorHandler(config, device)
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    print("checkpoint %s" % config["inference"]["checkpoint"], flush=True)

    transform = dataset_transform(
        config["data"]["image_size"], *config["data"]["shoeprint_norm"], random_image_flip=False
    )
    shoeprints = ShoeDataset(
        config["data"]["shoeprint_data_dir"],
        mode="train",
        transform=transform,
        channels=config["data"]["image_channels"],
    )
    random.seed(0)
    torch.manual_seed(0)
    prints = shoeprints.random_sample(39)
    main_print = prints[:1]

    for sub in ("styles", "consistency"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- candidate styles, then the maximally distinct subset -----------------
    z = torch.randn(N_CAND, config["architecture"]["s_dim"], device=device)
    with torch.no_grad():
        cand = handler.mapping_network(z).cpu()
    sel = maximin_styles(handler, main_print, cand, args.n_styles, device, lp)

    # order densest-first so a row reads as increasing degradation
    imgs = render(handler, sel, main_print)
    sel = sel[imgs.mean(dim=(1, 2, 3)).argsort()]
    imgs = render(handler, sel, main_print)

    save(main_print, args.out_dir / "styles" / "shoeprint.png")
    for i, im in enumerate(imgs, 1):
        save(im, args.out_dir / "styles" / ("shoemark_%d.png" % i))

    # --- one randomly sampled style across thirty-six shoeprints --------------
    # seed picked by eye from nine draws: a mid-density mark, visible in print
    g_style = torch.Generator().manual_seed(4)
    z_rand = torch.randn(1, config["architecture"]["s_dim"], generator=g_style).to(device)
    with torch.no_grad():
        w_rand = handler.mapping_network(z_rand).cpu()
    tiles = [render(handler, w_rand, prints[k:k + 1]) for k in range(3, 39)]
    grid = torchvision.utils.make_grid(torch.cat(tiles), nrow=6, padding=2,
                                       normalize=True, value_range=(-1, 1))
    torchvision.utils.save_image(grid, args.out_dir / "consistency" / "multiple_1.png")

    # --- the same styles on two further shoeprints ---------------------------
    for k in (1, 2):
        p = prints[k:k + 1]
        row = render(handler, sel, p)
        grid = torchvision.utils.make_grid(torch.cat([p, row]), nrow=args.n_styles + 1,
                                           padding=2, normalize=True, value_range=(-1, 1))
        torchvision.utils.save_image(grid, args.out_dir / "consistency" / ("single_%d.png" % k))

    print("wrote %s" % args.out_dir)


if __name__ == "__main__":
    main()
