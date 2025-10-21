"""Modified from https://github.com/sbarratt/inception-score-pytorch."""

import numpy as np
import torch
import torch.utils.data
from scipy.stats import entropy
from torch.nn import functional as F


def inception_score(imgs, inception_model, device, batch_size=32, splits=1):
    """Compute the inception score of the generated images."""
    # Set up dtype
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    n_imgs = len(imgs)
    preds = np.zeros((n_imgs, 1000))

    with torch.no_grad():
        for i, cpu_batch in enumerate(dataloader):
            batch = cpu_batch.to(device)

            preds[i * batch_size : (i + 1) * batch_size] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (n_imgs // splits) : (k + 1) * (n_imgs // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
