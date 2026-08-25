"""Score each run's best-FID checkpoint and record the canonical results row.

Selection comes from the per-checkpoint sweep (test_kid_fid.py), which seeds
its style draws per checkpoint (eval_seed, default 0), so the sweep's KID/FID
for the selected checkpoint are already the reportable numbers: the FID
recomputed here is bit-identical, and the KID differs only through the
estimator's random subset averaging, below reporting precision. What this
script adds is the seeded CIS for the selected checkpoint (test_cis.py sweeps
every checkpoint and is unseeded) and the bookkeeping: one row per run
appended to checkpoints/seeded_best_scores.csv, the file every reported
table cell traces back to. Runs already present are skipped, so an
interrupted invocation resumes.

    uv run python src/rescore_best.py                # all new_data_partition_fid* runs
    uv run python src/rescore_best.py --runs new_kl_loss_1 new_kl_loss_2
"""

import argparse
import csv
import re
import traceback
from pathlib import Path

import numpy as np
import torch
from torchvision.models.inception import inception_v3
from tqdm import tqdm

from one_to_many_gan.core.evaluation import validate_cis, validate_kid_fid
from one_to_many_gan.data.config import load_config
from one_to_many_gan.data.datasets import CyclingDataLoader, ShoeDataset, dataset_transform
from one_to_many_gan.model.builder import Generator, MappingNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.toml")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--runs",
    nargs="*",
    default=None,
    help="run directory names under checkpoints/ "
    "(default: every new_data_partition_fid* run)",
)
args = parser.parse_args()

config = load_config(args.config)

device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

torch.backends.fp32_precision = "tf32"
torch.backends.cuda.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"

# * Models

generator = Generator(
    input_nc=config["data"]["image_channels"],
    s_dim=config["architecture"]["s_dim"],
    image_size=config["data"]["image_size"],
    min_latent_resolution=config["architecture"]["min_latent_resolution"],
    n_resnet_blocks=config["architecture"]["n_resnet_blocks"],
)

mapping_network = MappingNetwork(
    features=config["architecture"]["s_dim"],
    n_layers=config["architecture"]["mapping_network_layers"],
    style_mixing_prob=0,
    n_gen_blocks=generator.n_style_blocks,
)

generator = generator.to(device).eval()
mapping_network = mapping_network.to(device).eval()
inception_model = inception_v3(weights="DEFAULT").to(device)
inception_model.eval()


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location=device)

    def strip_prefix(state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    generator.load_state_dict(strip_prefix(checkpoint["generator_state_dict"]))
    mapping_network.load_state_dict(strip_prefix(checkpoint["mapping_network_state_dict"]))


# * Data (mirrors test_kid_fid.py / test_cis.py: unshuffled, so the print
# order is deterministic and only the style draws needed seeding)

shoeprint_transform = dataset_transform(
    config["data"]["image_size"],
    *config["data"]["shoeprint_norm"],
    random_image_flip=False,
)

shoeprint_train_dataloader = torch.utils.data.DataLoader(
    ShoeDataset(
        config["data"]["shoeprint_data_dir"],
        mode="train",
        transform=shoeprint_transform,
        channels=config["data"]["image_channels"],
    ),
    batch_size=64,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)

shoeprint_val_dataloader = torch.utils.data.DataLoader(
    ShoeDataset(
        config["data"]["shoeprint_data_dir"],
        mode="val",
        transform=shoeprint_transform,
        channels=config["data"]["image_channels"],
    ),
    batch_size=100,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)


def best_step(run: Path) -> int:
    """Best-FID step from the run's sweep (last occurrence wins on re-sweeps)."""
    scores = {}
    for line in (run / "kid_fid_scores.txt").read_text().splitlines():
        m = re.match(r"Step (\d+) \| kid: [\d.eE+-]+ fid: ([\d.eE+-]+)", line)
        if m:
            scores[int(m.group(1))] = float(m.group(2))
    if not scores:
        raise RuntimeError("no parseable sweep lines in %s" % (run / "kid_fid_scores.txt"))
    return min(scores, key=scores.get)


def rescore(run: Path) -> dict:
    step = best_step(run)
    load_checkpoint(run / "models" / f"{step}.tar")

    torch.manual_seed(args.seed)
    with torch.no_grad():
        fid_score, kid_score = validate_kid_fid(
            config,
            device,
            CyclingDataLoader(shoeprint_train_dataloader),
            mapping_network,
            generator,
            run.parent / "val",
            config["data"]["shoemark_data_dir"] / "train",
        )

    torch.manual_seed(args.seed)
    shoeprints = next(iter(shoeprint_val_dataloader))
    cis_scores = []
    for shoeprint in tqdm(shoeprints, desc="CIS %s" % run.name, leave=False):
        with torch.no_grad():
            cis_scores.append(
                validate_cis(
                    config, device, shoeprint, mapping_network, generator, inception_model
                )
            )

    return {
        "run": run.name,
        "step": step,
        "fid": fid_score,
        "kid": kid_score,
        "cis": float(np.mean(cis_scores)),
        "eval_seed": args.seed,
    }


def main():
    root = config["training"]["checkpoint_directory"]
    runs = (
        [root / name for name in args.runs]
        if args.runs
        else sorted(p for p in root.glob("new_data_partition_fid*") if p.is_dir())
    )

    out = root / "seeded_best_scores.csv"
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {row["run"] for row in csv.DictReader(fh)}

    fields = ["run", "step", "fid", "kid", "cis", "eval_seed"]
    failed = []
    for run in runs:
        if run.name in done:
            print("%s already scored, skipping" % run.name, flush=True)
            continue
        try:
            row = rescore(run)
        except Exception:
            traceback.print_exc()
            failed.append(run.name)
            continue
        new = not out.exists()
        with out.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(row, flush=True)

    if failed:
        raise SystemExit("failed runs: %s" % ", ".join(failed))


if __name__ == "__main__":
    main()
