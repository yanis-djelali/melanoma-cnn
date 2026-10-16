import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.bloc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classificateur = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),   # 224 -> 112 -> 56 -> 28
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = self.classificateur(x)
        return x


from torchvision import models


def charger_resnet18_gele(num_classes):
    """Charge ResNet18 pre-entraine et gele tout sauf la derniere couche."""
    # Charger ResNet18 avec poids ImageNet
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Geler tout le backbone
    for param in resnet.parameters():
        param.requires_grad = False
    
    # Remplacer la derniere couche
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    
    return resnet

def compter_parametres(model):
    entrainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametres entrainables : {entrainables:,}")
    print(f"Parametres totaux       : {total:,}")
    print(f"Ratio entrainables/total : {entrainables/total:.4%}")
    return entrainables, total

def charger_resnet18_finetune(num_classes):
    """Charge ResNet18 pre-entraine, gele tout sauf layer4 + fc."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Geler tout le backbone
    for param in resnet.parameters():
        param.requires_grad = False
    
    # Remplacer la derniere couche
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    
    # Degeler layer4
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    
    return resnet

def charger_resnet18_finetune_layer3_4(num_classes):
    """Charge ResNet18 pre-entraine, gele tout sauf layer3 + layer4 + fc."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    for param in resnet.parameters():
        param.requires_grad = False
    
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    
    for param in resnet.layer3.parameters():
        param.requires_grad = True
    for param in resnet.layer4.parameters():
        param.requires_grad = True
    
    return resnet