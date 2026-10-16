"""Per-channel mean and standard deviation of the training set (used by Normalize).

    python -m scripts.compute_stats

Prints two estimates: the original one (average of per-batch statistics) and the exact one
(accumulated sums over every pixel). Only training images are used, never validation ones.
"""
import torch

from melanoma_cnn import transforms as T
from melanoma_cnn.config import TRAIN_DIR
from melanoma_cnn.data import make_loader


def main():
    loader = make_loader(TRAIN_DIR, T.BASE, shuffle=False, batch_size=64)
    batch_mean, batch_std = torch.zeros(3), torch.zeros(3)
    # float64 accumulators: about 6e8 pixels per channel would lose precision in float32
    pixel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_sq_sum = torch.zeros(3, dtype=torch.float64)
    n_pixels, n_batches = 0, 0

    for images, _ in loader:  # images: [B, 3, H, W]
        # dims 0, 2, 3 = batch, height, width -> one value per channel
        batch_mean += images.mean(dim=[0, 2, 3])
        batch_std += images.std(dim=[0, 2, 3])
        n_batches += 1
        pixel_sum += images.double().sum(dim=[0, 2, 3])
        pixel_sq_sum += (images.double() ** 2).sum(dim=[0, 2, 3])
        n_pixels += images.numel() // 3

    mean = pixel_sum / n_pixels
    std = (pixel_sq_sum / n_pixels - mean ** 2).sqrt()  # Var(X) = E[X^2] - E[X]^2
    print(f"Original estimate (mean of batch stats) : mean {(batch_mean / n_batches).tolist()}, "
          f"std {(batch_std / n_batches).tolist()}")
    print(f"Exact (all pixels)                     : mean {mean.tolist()}, std {std.tolist()}")


if __name__ == "__main__":
    main()
