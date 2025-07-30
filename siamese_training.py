"""Train a Siamese model using images generated on the fly."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.data.config import load_config
from src.data.datasets import LabeledCombinedDataset, dataset_transform
from src.model.siamese import SharedSiamese
from tqdm import tqdm

config = load_config("config.toml")

model_size = "s"
pre_trained = False

p_val = 2
margin = 0.5
batch_size = 32


def seed_worker(worker_id):
    """Seed DataLoader workers with random seed."""
    worker_seed = (
        config["training"]["random_seed"] + worker_id
    ) % 2**32  # Ensure we don't overflow 32 bit
    np.random.default_rng(worker_seed)
    random.seed(worker_seed)

    # Passed to dataloaders
    dataloader_g = torch.Generator()
    dataloader_g.manual_seed(config["training"]["random_seed"])


torch.manual_seed(config["training"]["random_seed"])
np.random.default_rng(config["training"]["random_seed"])
random.seed(config["training"]["random_seed"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SharedSiamese().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)


random.seed(config["training"]["random_seed"])

dataset = LabeledCombinedDataset(
    config["data"]["shoeprint_data_dir"],
    config["data"]["shoemark_data_dir"],
    mode="train",
    shoeprint_transform=dataset_transform(config["data"]["image_size"]),
    shoemark_transform=dataset_transform(config["data"]["image_size"], offset=True),
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    drop_last=False,
    worker_init_fn=seed_worker,
    persistent_workers=True,
)


# Find negatives closer to the anchor than positives
# Violating d(anchor, positive) + margin < d(anchor, negative)
def training_loop(epochs: int, print_iter: int, save_iter: int):
    """Run training loop for siamese model."""
    with tqdm(total=(epochs * len(dataset)) // batch_size, dynamic_ncols=True) as pbar:
        val = test(
            "/home/struan/Datasets/WVU2019 Cropped/Gallery",
            "/home/struan/Datasets/WVU2019 Cropped/Query",
        )
        #        val = validate()
        line = f"Validation: p5 = {val}\n"
        pbar.write(line)
        with open("siamese/siamese.log", "a") as f:
            f.write(line)

        for epoch in range(epochs):
            pbar.set_description(f"Epoch: {epoch}")
            losses = 0
            avg_size = 0

            for step, (shoeprint_batch, shoemark_batch) in enumerate(loader):
                shoeprints = shoeprint_batch.to(device)
                shoemarks = shoemark_batch.to(device)

                # Get embeddings
                anchors = model(shoeprints)  # [b, d]

                positives = model(shoemarks.flatten(0, 1)).unflatten(
                    0, (shoemarks.shape[0], 5)
                )  # [b, 5, d]
                anchors = F.normalize(anchors, p=2, dim=1)  # L2-normalise
                positives = F.normalize(positives, p=2, dim=2)

                # Squared L2 distances: anchors vs all positives (across batch)
                anchors_exp = anchors.unsqueeze(1)  # [b, 1, d]
                positives_exp = positives.flatten(0, 1).unsqueeze(0)  # [1, b*5, d]
                dist_matrix = (anchors_exp - positives_exp).pow(2).sum(dim=2)  # b, b*5

                # Mask for valid triplets (anchor vs other identities' positives)
                identity_mask = torch.eye(anchors.shape[0], device=device)  # [b, b]
                identity_mask = identity_mask.repeat_interleave(5, dim=1)  # [b, b*5]
                neg_mask = ~identity_mask.bool()  # Take negatives from different shoes

                batch_losses = torch.tensor(0.0).to(device)
                count = 0

                for i in range(anchors.shape[0]):
                    # For current anchor, get its 5 positives
                    pos_start, pos_end = i * 5, (i + 1) * 5
                    d_ap = dist_matrix[
                        i, pos_start:pos_end
                    ]  # [5] distances to its positives

                    for d_pos in d_ap:
                        # Semi-hard condition: d_pos < d_an < d_pos + alpha
                        valid_negs = (
                            (dist_matrix[i] > d_pos)
                            & (dist_matrix[i] < d_pos + margin)
                            & neg_mask[i]
                        )
                        if not valid_negs.any():
                            continue  # Skip if no valid semi-hard triplets

                        # Find closest negative in the valid band
                        d_semi_hard = dist_matrix[i, valid_negs].min()
                        batch_losses += F.relu(d_pos - d_semi_hard + margin)
                        count += 1

                loss = (
                    batch_losses / count
                    if count > 0
                    else torch.tensor(0.0, device=device, requires_grad=True)
                )
                losses += loss.item()
                avg_size += count

                if step % print_iter == 0 and step != 0:
                    line = f"{(losses / print_iter)} | avg batch size {avg_size / print_iter}\n"
                    pbar.write(line)
                    with open("siamese/siamese.log", "a") as f:
                        f.write(line)
                    losses = 0
                    avg_size = 0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update()

            val = test(
                "/home/struan/Datasets/WVU2019 Cropped/Gallery",
                "/home/struan/Datasets/WVU2019 Cropped/Query",
            )

            # val = validate()
            line = f"Epoch {epoch} validation: p5 = {val}\n"
            pbar.write(line)
            with open("siamese/siamese.log", "a") as f:
                f.write(line)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                },
                f"siamese/siamese_{epoch}.tar",
            )


@torch.no_grad()
def validate(p: int = 5):
    """Test saved model."""
    model.eval()
    dataset = LabeledCombinedDataset(
        config["data"]["shoeprint_data_dir"],
        config["data"]["shoemark_data_dir"],
        mode="val",
        shoeprint_transform=dataset_transform(config["data"]["image_size"]),
        shoemark_transform=dataset_transform(config["data"]["image_size"], offset=True),
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    # Process in batches to save memory
    all_p_embs, all_m_embs = [], []
    for shoeprint, shoemark in val_dataloader:
        all_p_embs.append(model(shoeprint.to(device)).cpu())
        all_m_embs.append(model(shoemark.to(device)).cpu())

    model.train()

    shoemark_embeddings = torch.cat(all_m_embs)  # [N, d]
    shoeprint_embeddings = torch.cat(all_p_embs)  # [N, d]

    # Pairwise distances matrix [N, N]
    dists = torch.cdist(shoeprint_embeddings, shoemark_embeddings, p=p_val)

    # Get ranks (position of correct match in sorted distances)
    ranks = (dists.diag().unsqueeze(1) >= dists).sum(dim=1).float()

    # Calculate top-p% accuracy
    k = max(1, int(len(dataset) * p / 100))

    return (ranks <= k).float().mean().item()


@torch.no_grad()
def test(shoeprint_path: str | Path, shoemark_path: str | Path):
    model.eval()

    # checkpoint = torch.load("siamese/siamese_235.tar")

    # model.load_state_dict(checkpoint["model_state_dict"])

    shoeprint_path = Path(shoeprint_path)
    shoemark_path = Path(shoemark_path)

    shoeprint_files = list(shoeprint_path.rglob("*.png"))
    shoemark_files = list(shoemark_path.rglob("*.png"))

    transform = dataset_transform(config["data"]["image_size"], offset=False)

    def calc_embedding(f: Path):
        i = Image.open(f)
        t = transform(i).to(device)
        return model(t.unsqueeze(0)).cpu()

    shoeprint_embeddings = {
        f.stem[:3]: calc_embedding(f).squeeze() for f in shoeprint_files
    }
    shoemark_embeddings = {f.stem: calc_embedding(f).squeeze() for f in shoemark_files}

    ranks = []
    shoeprint_embs = torch.stack(list(shoeprint_embeddings.values()))

    for shoe_id, shoemark_embedding in shoemark_embeddings.items():
        dists = torch.cdist(shoemark_embedding.unsqueeze(0), shoeprint_embs, p=p_val)
        dists = dists.squeeze()
        sorted = torch.argsort(dists)

        shoe_id = shoe_id[:3]

        correct_idx = list(shoeprint_embeddings.keys()).index(shoe_id)
        rank = (sorted == int(correct_idx)).nonzero().item()

        ranks.append(rank)

    ranks = np.array(ranks)

    # p=5
    return np.mean(ranks <= 10)


if __name__ == "__main__":
    # checkpoint = torch.load("siamese.bkp/siamese_225.tar")

    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optim_state_dict"])

    training_loop(500, 100, 5)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        "siamese/siamese_final.tar",
    )
# print(
#     test(
#         "/home/struan/Datasets/WVU2019 Cropped/Gallery",
#         "/home/struan/Datasets/WVU2019 Cropped/Query",
#     )
# )
