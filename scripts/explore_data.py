"""Figures of parts 1 to 3: examples per class, class distribution, RGB channels, one batch
of the training set and one batch of the validation set.

    python -m scripts.explore_data
"""
import torch

from melanoma_cnn import plots
from melanoma_cnn import transforms as T
from melanoma_cnn.config import FIGURES_DIR, SEED, TRAIN_DIR, VAL_DIR
from melanoma_cnn.data import MelanomaDataset, make_loader

OUT_DIR = FIGURES_DIR / "exploration"
IMAGES_PER_CLASS = 6  # the brief asks for a grid of at least 12 images


def examples_per_class(dataset):
    images, titles = [], []
    for label, name in enumerate(dataset.classes):
        indices = [i for i, (_, y) in enumerate(dataset.samples) if y == label][:IMAGES_PER_CLASS]
        for i in indices:
            images.append(dataset[i][0].permute(1, 2, 0).numpy())
            titles.append(name)
    plots.image_grid(images, titles, "Exemples par classe (train)", OUT_DIR / "exemples_images.png",
                     columns=IMAGES_PER_CLASS)


def class_distribution(dataset):
    counts = {name: sum(1 for _, y in dataset.samples if y == label)
              for label, name in enumerate(dataset.classes)}
    print(f"Train : {counts}")
    plots.bar_counts(counts, "Distribution des classes (train)", OUT_DIR / "distribution_classes.png")


def rgb_channels(dataset):
    image = dataset[0][0]  # [3, H, W], values in [0, 1]
    print(f"Tensor shape {list(image.shape)}, min {image.min():.4f}, max {image.max():.4f}")
    panels = [image.permute(1, 2, 0).numpy()] + [image[c].numpy() for c in range(3)]
    plots.image_grid(panels, ["Image", "Canal R", "Canal V", "Canal B"],
                     f"Forme du tenseur : {list(image.shape)}", OUT_DIR / "canaux_rgb.png",
                     cmaps=[None, "Reds", "Greens", "Blues"])


def batch_grid(root, shuffle, title, filename):
    loader = make_loader(root, T.BASE, shuffle=shuffle)
    images, labels = next(iter(loader))
    print(f"{title} : batch {list(images.shape)}, labels {list(labels.shape)}")
    classes = loader.dataset.classes
    plots.image_grid([img.permute(1, 2, 0).numpy() for img in images[:8]],
                     [classes[y] for y in labels[:8].tolist()], title, OUT_DIR / filename)


def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_set = MelanomaDataset(TRAIN_DIR, T.BASE)
    print(f"Train : {len(train_set)} images, val : {len(MelanomaDataset(VAL_DIR))} images")
    examples_per_class(train_set)
    class_distribution(train_set)
    rgb_channels(train_set)
    batch_grid(TRAIN_DIR, True, "Batch du train (shuffle=True)", "batch_train.png")
    batch_grid(VAL_DIR, False, "Batch de la validation (shuffle=False)", "batch_val.png")


if __name__ == "__main__":
    main()
