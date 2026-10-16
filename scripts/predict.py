"""Demo: class probabilities for one image with a trained model.

    python -m scripts.predict resnet_ft_layer34 path/to/image.jpg

Student project, not a medical tool. The model was trained on dermoscopic images
(acquired with a dermatoscope); on an ordinary photo its output has no meaning.
"""
import argparse

import torch
from PIL import Image

from melanoma_cnn import transforms as T
from melanoma_cnn.experiments import EXPERIMENTS
from scripts.analyze_errors import load_model

CLASSES = ["Benign", "Malignant"]  # sorted folder names of the training set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("image")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.experiment, len(CLASSES), device).eval()
    image = T.PREDICT(Image.open(args.image).convert("RGB")).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        probs = torch.softmax(model(image.to(device)), dim=1)[0].tolist()
    for name, p in zip(CLASSES, probs):
        print(f"{name:<10} {100 * p:5.1f} %")
    print("Projet etudiant, pas un outil medical.")


if __name__ == "__main__":
    main()
