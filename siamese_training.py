"""Train a Siamese model using images generated on the fly."""

import random

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.data.config import load_config
from src.data.datasets import LabeledCombinedDataset, dataset_transform
from src.model.siamese import SharedSiamese

config = load_config("config.toml")

model_size = "s"
pre_trained = False

p_val = 2
margin = 0.5

triplet_loss = nn.TripletMarginLoss(margin=margin, p=p_val, swap=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SharedSiamese().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

batch_size = 16

random.seed(config["training"]["random_seed"])

dataset = LabeledCombinedDataset(
    config["data"]["shoeprint_data_dir"],
    config["data"]["shoemark_data_dir"],
    mode="train",
    transform=dataset_transform(config["data"]["image_size"]),
)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)


# Find negatives closer to the anchor than positives
# Violating d(anchor, positive) + margin < d(anchor, negative)
def training_loop(epochs: int, print_iter: int):
    """Run training loop for siamese model."""
    losses = 0
    avg_size = 0

    with tqdm(total=(epochs * len(dataset)) // batch_size) as pbar:
        val = validate()
        pbar.write(f"Validation: p5 = {val}")
        for epoch in range(epochs):
            pbar.set_description(f"Epoch: {epoch}")

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
                    d_ap = dist_matrix[i, pos_start:pos_end]  # [5] distances to its positives

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

                loss = batch_losses / count if count > 0 else torch.tensor(0.0)
                losses += loss.item()
                avg_size += count

                if step % print_iter == 0 and step != 0:
                    pbar.write(f"{(losses / print_iter)} | avg batch size {avg_size / print_iter}")
                    losses = 0
                    avg_size = 0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update()

            val = validate()
            pbar.write(f"Validation: p5 = {val}")


@torch.no_grad()
def validate(p: int = 5):
    """Test saved model."""
    model.eval()
    dataset = LabeledCombinedDataset(
        config["data"]["shoeprint_data_dir"],
        config["data"]["shoemark_data_dir"],
        mode="val",
        transform=dataset_transform(config["data"]["image_size"]),
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
