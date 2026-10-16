"""SimpleCNN trained from scratch and ResNet18 adapted by transfer learning."""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Three Conv-ReLU-MaxPool blocks then a two-layer classifier, for 224x224 RGB inputs."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16),   # 224 -> 112
            conv_block(16, 32),  # 112 -> 56
            conv_block(32, 64),  # 56 -> 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def resnet18_classifier(num_classes, trainable_blocks=(), pretrained=True):
    """ResNet18 with a new head; every pre-trained weight is frozen except `trainable_blocks`.

    trainable_blocks=()                  -> frozen backbone, only the head is trained
    trainable_blocks=("layer4",)         -> partial fine-tuning
    trainable_blocks=("layer3", "layer4")
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)  # new layer: trainable by default
    for block in trainable_blocks:
        for param in getattr(net, block).parameters():
            param.requires_grad = True
    return net


def count_parameters(model):
    """Return (trainable, total) numbers of parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
