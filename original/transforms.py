import torchvision.transforms as transforms

MEAN = [0.7228, 0.5555, 0.5390]
STD  = [0.1886, 0.1974, 0.2110]

# Pipeline de base (Partie 5) : juste ToTensor
transform_base = transforms.Compose([
    transforms.ToTensor(),
])

# Pipeline avec normalisation (Partie 6)
transform_normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Pipeline avec augmentation pour le TRAIN (Partie 7)
train_transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Pipeline pour la VAL (pas d'augmentation, juste normalisation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])