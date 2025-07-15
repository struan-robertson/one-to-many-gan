"""Train a Siamese model using images generated on the fly."""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from src.core.generate import GeneratorHandler
from src.data.config import load_config
from src.data.datasets import ShoeDataset, dataset_transform
from src.model.siamese import SharedSiamese

config = load_config("config.toml")

model_size = "s"
pre_trained = False

p_val = 2
margin = 2

triplet_loss = nn.TripletMarginLoss(margin=margin, p=p_val, swap=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SharedSiamese().to(device)

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)

batch_size = 32

generator = GeneratorHandler(config, device, batch_size)


# Find negatives closer to the anchor than positives
# Violating d(anchor, positive) + margin < d(anchor, negative)
def training_loop(steps: int, print_iter: int):
    """Run training loop for siamese model."""
    previous_positives: torch.Tensor | None = None

    losses = 0
    avg_size = 0

    for step in trange(steps):
        shoeprints, shoemarks1, shoemarks2 = generator.generate(1)

        shoeprints = shoeprints.expand(batch_size, 3, 512, 256)
        shoemarks1 = shoemarks1.expand(batch_size, 3, 512, 256)
        shoemarks2 = shoemarks2.expand(batch_size, 3, 512, 256)

        previous_positives = shoemarks1

        anchor = model(shoeprints)
        positive1 = model(shoemarks1)
        positive2 = model(shoemarks2)
        negative = (
            model(previous_positives)
            if previous_positives is not None
            else model(generator.generate(1)[1].expand(batch_size, 3, 512, 256))
        )

        anchor_positive1_dist = torch.norm(anchor - positive1, p=p_val, dim=1)
        anchor_positive2_dist = torch.norm(anchor - positive2, p=p_val, dim=1)
        anchor_negative_dist = torch.norm(anchor - negative, p=p_val, dim=1)

        anchors = []
        positives = []
        negatives = []
        for i in range(anchor.shape[0]):
            if (
                anchor_positive1_dist[i]
                < anchor_negative_dist[i]
                < anchor_positive2_dist[i] + margin
            ):
                anchors.append(anchor[i])
                positives.append(positive1[i])
                negatives.append(negative[i])
            elif (
                anchor_positive2_dist[i]
                < anchor_negative_dist[i]
                < anchor_positive1_dist[i] + margin
            ):
                anchors.append(anchor[i])
                positives.append(positive2[i])
                negatives.append(negative[i])

        if len(anchors) == 0:
            continue

        avg_size += len(anchors)

        anchors = torch.stack(anchors, dim=0)
        positives = torch.stack(positives, dim=0)
        negatives = torch.stack(negatives, dim=0)

        output = triplet_loss(anchors, positives, negatives)

        losses += output.item()

        if step % print_iter == 0 and step != 0:
            tqdm.write(f"{(losses / print_iter)} | avg batch size {avg_size / print_iter}")
            losses = 0
            avg_size = 0

        output.backward()
        optimizer.step()


def test():
    """Test saved model."""
    with torch.no_grad():
        transform = dataset_transform(config["data"]["image_size"])
        shoeprint_data = ShoeDataset(
            config["data"]["shoeprint_data_dir"], mode="val", transform=transform
        )
        shoeprint_dataloader = torch.utils.data.DataLoader(
            shoeprint_data,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )

        shoeprints = list(shoeprint_dataloader)
        shoemarks = [
            generator.generate_from_shoeprint(shoeprint.to(device), 0.3).detach().cpu()
            for shoeprint in shoeprints
        ]

        shoeprint_embeddings = [
            model(shoeprint.to(device).expand(1, 3, 512, 256)).detach().cpu().squeeze()
            for shoeprint in shoeprints
        ]
        shoemark_embeddings = [
            model(shoemark.to(device).expand(1, 3, 512, 256)).detach().cpu().squeeze()
            for shoemark in shoemarks
        ]
        shoemark_embeddings = torch.stack(shoemark_embeddings)

        ranks = []

        for i, shoeprint_embedding in enumerate(shoeprint_embeddings):
            diff = shoemark_embeddings - shoeprint_embedding.unsqueeze(0)
            norms = torch.norm(diff, p=p_val, dim=1)

            sorted_indices = torch.argsort(norms)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()

            print(rank)

            ranks.append(rank)

        print(f"Mean rank: {np.mean(ranks)}")
