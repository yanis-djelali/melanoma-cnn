"""Matplotlib figures. No torch import: images arrive as numpy arrays [H, W, C] in [0, 1]."""
import matplotlib

matplotlib.use("Agg")  # write files, never open a window (also avoids X11 errors over SSH)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# First slots of a colorblind-checked categorical palette, always used in this order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
METRIC_NAMES = {"val_acc": "Accuracy validation", "val_loss": "Loss validation",
                "train_acc": "Accuracy train", "train_loss": "Loss train"}

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.8, "lines.linewidth": 2, "font.size": 10,
    "legend.frameon": False,
})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {path}")


def epochs_of(values):
    return range(1, len(values) + 1)


def epoch_axis(ax):
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def learning_curves(history, title, path):
    """Loss and accuracy, train vs validation, side by side."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, name in ((ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")):
        ax.plot(epochs_of(history[f"train_{metric}"]), history[f"train_{metric}"],
                color=COLORS[0], label="Train")
        ax.plot(epochs_of(history[f"val_{metric}"]), history[f"val_{metric}"],
                color=COLORS[1], label="Validation")
        epoch_axis(ax)
        ax.set_ylabel(name)
        ax.set_title(f"{name} : {title}", color=TEXT)
        ax.legend()
    save(fig, path)


def compare_runs(histories, metric, title, path, ylim=None):
    """One line per run for a given metric (e.g. 'val_acc'). `histories` maps label -> history."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for color, (label, history) in zip(COLORS, histories.items()):
        ax.plot(epochs_of(history[metric]), history[metric], color=color, label=label)
    epoch_axis(ax)
    ax.set_ylabel(METRIC_NAMES.get(metric, metric))
    ax.set_title(title, color=TEXT)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    save(fig, path)


def learning_rate(lrs, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(epochs_of(lrs), lrs, color=COLORS[0], marker="o", markersize=5)
    ax.set_yscale("log")
    epoch_axis(ax)
    ax.set_ylabel("Learning rate")
    ax.set_title("StepLR : division par 10 toutes les 7 epochs", color=TEXT)
    save(fig, path)


def confusion_matrix(matrix, classes, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Vrai label")
    ax.set_title(title, color=TEXT)
    top = max(max(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                    color="white" if value > top / 2 else TEXT)
    save(fig, path)


def image_grid(images, titles, suptitle, path, columns=4, cmaps=None):
    """`cmaps` (optional): one colormap per image, used for single-channel [H, W] arrays."""
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows), squeeze=False)
    cmaps = cmaps or [None] * len(images)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title, cmap in zip(axes.flat, images, titles, cmaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=9, color=TEXT)
    fig.suptitle(suptitle, color=TEXT)
    save(fig, path)


def bar_counts(counts, title, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(list(counts), list(counts.values()), color=COLORS[:len(counts)], width=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Nombre d'images")
    ax.set_title(title, color=TEXT)
    for i, value in enumerate(counts.values()):
        ax.text(i, value, str(value), ha="center", va="bottom", color=TEXT)
    save(fig, path)
