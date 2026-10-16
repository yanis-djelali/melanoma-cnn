"""Architecture of SimpleCNN and trainable / total parameters of every configuration.

    python -m scripts.model_summary
"""
import torch

from melanoma_cnn.experiments import EXPERIMENTS, build_model
from melanoma_cnn.models import SimpleCNN, count_parameters


def main():
    cnn = SimpleCNN(num_classes=2)
    print(cnn)
    x = torch.zeros(1, 3, 224, 224)
    for name, layer in cnn.features.named_children():
        x = layer(x)
        print(f"after block {name} : {list(x.shape)}")
    print(f"\n{'configuration':<20} {'trainable':>12} {'total':>12}")
    for name in EXPERIMENTS:
        trainable, total = count_parameters(build_model(name, 2, pretrained=False))
        print(f"{name:<20} {trainable:>12,} {total:>12,}")


if __name__ == "__main__":
    main()
