"""The nine configurations compared in the project, one variable changed at a time.

All CNN runs use the same SimpleCNN; all ResNet runs use Adam with lr=1e-3 on the new head
and lr=1e-4 on the unfrozen pre-trained blocks (smaller steps to keep ImageNet features).
"""
import torch.optim as optim

from melanoma_cnn import transforms as T
from melanoma_cnn.models import SimpleCNN, resnet18_classifier

EXPERIMENTS = {
    "cnn_simple":        {"model": "cnn", "pipeline": "base", "optimizer": "adam", "lr": 1e-3},
    "cnn_norm":          {"model": "cnn", "pipeline": "normalized", "optimizer": "adam", "lr": 1e-3},
    "cnn_norm_aug":      {"model": "cnn", "pipeline": "augmented", "optimizer": "adam", "lr": 1e-3},
    "cnn_adam_lr1e-4":   {"model": "cnn", "pipeline": "augmented", "optimizer": "adam", "lr": 1e-4},
    "cnn_sgd":           {"model": "cnn", "pipeline": "augmented", "optimizer": "sgd", "lr": 1e-2},
    "cnn_steplr":        {"model": "cnn", "pipeline": "augmented", "optimizer": "adam", "lr": 1e-3,
                          "step_lr": True},
    "resnet_frozen":     {"model": "resnet18", "pipeline": "augmented", "blocks": ()},
    "resnet_ft_layer4":  {"model": "resnet18", "pipeline": "augmented", "blocks": ("layer4",)},
    "resnet_ft_layer34": {"model": "resnet18", "pipeline": "augmented", "blocks": ("layer3", "layer4")},
}

PIPELINES = {  # (train transform, validation transform)
    "base": (T.BASE, T.BASE),
    "normalized": (T.NORMALIZED, T.NORMALIZED),
    "augmented": (T.TRAIN_AUGMENTED, T.NORMALIZED),
}


def build_model(name, num_classes, pretrained=True):
    config = EXPERIMENTS[name]
    if config["model"] == "cnn":
        return SimpleCNN(num_classes)
    return resnet18_classifier(num_classes, config["blocks"], pretrained=pretrained)


def build_optimizer(name, model):
    config = EXPERIMENTS[name]
    if config["model"] == "resnet18":
        groups = [{"params": getattr(model, block).parameters(), "lr": 1e-4}
                  for block in config["blocks"]]
        groups.append({"params": model.fc.parameters(), "lr": 1e-3})
        return optim.Adam(groups)
    if config["optimizer"] == "sgd":
        return optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    return optim.Adam(model.parameters(), lr=config["lr"])


def build_scheduler(name, optimizer):
    """StepLR divides the learning rate by 10 every 7 epochs (1e-3, 1e-4, 1e-5)."""
    if EXPERIMENTS[name].get("step_lr"):
        return optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return None


def transforms_for(name):
    return PIPELINES[EXPERIMENTS[name]["pipeline"]]
