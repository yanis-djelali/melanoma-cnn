"""Confusion matrix, per-class report and examples of errors of a trained model.

    python -m scripts.analyze_errors resnet_ft_layer34

Needs checkpoints/<experiment>.pt (written by scripts.train). Examples are drawn at random,
with a fixed seed, separately for each true class: the validation set is sorted by class,
so taking the first 8 errors would only show one kind of error.
"""
import argparse
import json
import random

import torch
from sklearn.metrics import classification_report, confusion_matrix

from melanoma_cnn import plots
from melanoma_cnn import transforms as T
from melanoma_cnn.config import CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR, SEED, VAL_DIR
from melanoma_cnn.data import make_loader
from melanoma_cnn.engine import predict
from melanoma_cnn.experiments import EXPERIMENTS, build_model, transforms_for

PER_CLASS = 4  # 4 + 4 = 8 errors and 8 successes, as in the brief


def load_model(experiment, num_classes, device):
    model = build_model(experiment, num_classes, pretrained=False)  # weights come from the file
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{experiment}.pt", map_location=device))
    return model.to(device)


def sample(y_true, y_pred, correct, rng):
    """Indices of up to PER_CLASS images per true class, correctly classified or not."""
    chosen = []
    for label in sorted(set(y_true)):
        candidates = [i for i, (t, p) in enumerate(zip(y_true, y_pred))
                      if t == label and (t == p) == correct]
        chosen += rng.sample(candidates, min(PER_CLASS, len(candidates)))
    return chosen


def save_examples(dataset, indices, y_pred, probs, title, path):
    images, titles = [], []
    for i in indices:
        image, label = dataset[i]
        images.append(T.to_displayable(image))
        predicted = y_pred[i]
        titles.append(f"Vrai : {dataset.classes[label]}\n"
                      f"Predit : {dataset.classes[predicted]} (p={probs[i][predicted]:.2f})")
    plots.image_grid(images, titles, title, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = make_loader(VAL_DIR, transforms_for(args.experiment)[1], shuffle=False)
    dataset = loader.dataset
    model = load_model(args.experiment, len(dataset.classes), device)
    y_true, y_pred, probs = (t.tolist() for t in predict(model, loader, device))

    matrix = confusion_matrix(y_true, y_pred).tolist()
    print(f"Confusion matrix (rows = true, columns = predicted) : {matrix}")
    print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=3))
    out = RESULTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=dataset.classes, output_dict=True)
    (out / f"{args.experiment}.json").write_text(
        json.dumps({"classes": dataset.classes, "matrix": matrix, "report": report}, indent=1),
        encoding="utf-8")

    fig_dir = FIGURES_DIR / "analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plots.confusion_matrix(matrix, dataset.classes, args.experiment,
                           fig_dir / f"matrice_confusion_{args.experiment}.png")
    rng = random.Random(SEED)
    save_examples(dataset, sample(y_true, y_pred, False, rng), y_pred, probs,
                  "Exemples mal classes", fig_dir / f"erreurs_{args.experiment}.png")
    save_examples(dataset, sample(y_true, y_pred, True, rng), y_pred, probs,
                  "Exemples bien classes", fig_dir / f"succes_{args.experiment}.png")


if __name__ == "__main__":
    main()
