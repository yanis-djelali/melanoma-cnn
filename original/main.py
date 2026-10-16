import os
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset import MelonaDataset

# --------------------------------------------------
# Configuration du device (GPU si disponible)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilise : {device}")

TRAIN_DIR = "melanoma-cancer-dataset/train"
classes = sorted(os.listdir(TRAIN_DIR))
num_classes = len(classes)
print(f"\nNombre de classes : {num_classes}")
print(f"Classes : {classes}")
counts = {}
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len(os.listdir(chemin_classe))
    counts[classe] = nb_images
    print(f"  {classe} : {nb_images} images")

#  Afficher 2 images par classe dans une grille 
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

i = 0
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    fichiers = sorted(os.listdir(chemin_classe))[:2]
    for fichier in fichiers:
        chemin_image = os.path.join(chemin_classe, fichier)
        img = Image.open(chemin_image)
        axes[i].imshow(img)
        axes[i].set_title(classe)
        axes[i].axis("off")
        i += 1

plt.suptitle("Exemples d'images par classe")
plt.tight_layout()
plt.savefig("exemples_images.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n>>> Figure sauvegardee : exemples_images.png")

# --- Diagramme en barres ---
plt.figure(figsize=(8, 5))
plt.bar(counts.keys(), counts.values(), color=['steelblue', 'tomato'])
plt.xlabel("Classe")
plt.ylabel("Nombre d'images")
plt.title("Distribution des classes dans le train set")
plt.tight_layout()
plt.savefig("distribution_classes.png", dpi=150)
plt.close()
print(">>> Figure sauvegardee : distribution_classes.png")

dossier_benign = os.path.join(TRAIN_DIR, "Benign")
premier_fichier = sorted(os.listdir(dossier_benign))[0]
chemin = os.path.join(dossier_benign, premier_fichier)

img_pil = Image.open(chemin).convert("RGB")
print(f"\nTaille originale (PIL) : {img_pil.size}")
print(f"Type des pixels PIL : {type(img_pil.getpixel((0,0)))}")

# --- Convertir en tenseur PyTorch ---
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_pil)

print(f"Forme du tenseur : {img_tensor.shape}")
print(f"Valeur min : {img_tensor.min():.4f}")
print(f"Valeur max : {img_tensor.max():.4f}")

# --- Afficher l'image originale + les 3 canaux RGB ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes = axes.flatten()

# Image originale (reconstruite depuis le tenseur)
# On réordonne [C, H, W] en [H, W, C] pour matplotlib
img_display = img_tensor.permute(1, 2, 0).numpy()
axes[0].imshow(img_display)
axes[0].set_title("Image originale")
axes[0].axis("off")

# Canal Rouge (indice 0)
axes[1].imshow(img_tensor[0].numpy(), cmap='Reds')
axes[1].set_title("Canal Rouge")
axes[1].axis("off")

# Canal Vert (indice 1)
axes[2].imshow(img_tensor[1].numpy(), cmap='Greens')
axes[2].set_title("Canal Vert")
axes[2].axis("off")

# Canal Bleu (indice 2)
axes[3].imshow(img_tensor[2].numpy(), cmap='Blues')
axes[3].set_title("Canal Bleu")
axes[3].axis("off")

plt.suptitle(f"Forme du tenseur : {list(img_tensor.shape)}")
plt.tight_layout()
plt.savefig("canaux_rgb.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : canaux_rgb.png")

# ==================================================
# Partie 3 : Dataset et DataLoader
# ==================================================

# --- Transformation de base : juste convertir en tenseur ---
transform_base = transforms.Compose([
    transforms.ToTensor(),
])

# --- Créer les datasets (test joue le role de validation) ---
train_dataset = MelonaDataset("melanoma-cancer-dataset/train", transform=transform_base)
val_dataset = MelonaDataset("melanoma-cancer-dataset/test", transform=transform_base)

print(f"\nTaille du train set : {len(train_dataset)}")
print(f"Taille du val set : {len(val_dataset)}")
print(f"Classes : {train_dataset.classes}")
print(f"Mapping classe -> entier : {train_dataset.class_to_idx}")

# --- Créer les DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6)

# --- Verifier la forme d'un batch ---
images_batch, labels_batch = next(iter(train_loader))
print(f"Forme d'un batch d'images : {images_batch.shape}")
print(f"Forme des labels : {labels_batch.shape}")

# --- Visualiser 8 images du batch ---
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i in range(8):
    img = images_batch[i].permute(1, 2, 0).numpy()
    label_nom = train_dataset.classes[labels_batch[i].item()]
    axes[i].imshow(img)
    axes[i].set_title(label_nom)
    axes[i].axis("off")

plt.suptitle("Batch d'images du train set")
plt.tight_layout()
plt.savefig("batch_train.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : batch_train.png")
# ==================================================
# Partie 5 : Construire le modèle
# ==================================================
from model import SimpleCNN, compter_parametres

num_classes = len(train_dataset.classes)  # devrait être 2

model = SimpleCNN(num_classes=num_classes).to(device)
print(f"\nModele cree : SimpleCNN avec {num_classes} classes")
compter_parametres(model)

# --- Test : faire passer un batch à travers le modèle ---
images_test, labels_test = next(iter(train_loader))
images_test = images_test.to(device)
outputs = model(images_test)
print(f"Forme d'entree : {images_test.shape}")
print(f"Forme de sortie : {outputs.shape}")

# ==================================================
# Partie 5 : Entrainement du CNN
# ==================================================
from train import train_one_epoch, evaluate

NUM_EPOCHS = 20

# Critere de loss et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Historique pour tracer les courbes
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

LANCER_TRAIN_PARTIE_5 = False

if LANCER_TRAIN_PARTIE_5:
    print("\n=== Debut de l'entrainement ===\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        duree = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")
    print("\n=== Entrainement termine ===")
else:
    history["train_loss"] = [0.4042, 0.3460, 0.3321, 0.3169, 0.3088, 0.2960, 0.2833, 0.2773, 0.2678, 0.2563, 0.2503, 0.2266, 0.2170, 0.2032, 0.1975, 0.1778, 0.1754, 0.1570, 0.1565, 0.1419]
    history["val_loss"]   = [0.4010, 0.3692, 0.3551, 0.2988, 0.2808, 0.3143, 0.2778, 0.2927, 0.2966, 0.2520, 0.2478, 0.3094, 0.3063, 0.2240, 0.2385, 0.2229, 0.2293, 0.2603, 0.2372, 0.2233]
    history["train_acc"]  = [0.819, 0.847, 0.858, 0.861, 0.865, 0.875, 0.883, 0.884, 0.887, 0.894, 0.896, 0.908, 0.910, 0.917, 0.919, 0.926, 0.931, 0.938, 0.938, 0.945]
    history["val_acc"]    = [0.821, 0.887, 0.848, 0.899, 0.896, 0.847, 0.898, 0.879, 0.894, 0.918, 0.913, 0.871, 0.868, 0.928, 0.916, 0.925, 0.927, 0.918, 0.927, 0.936]
    print("\n=== Train Partie 5 SKIPPE ===")

# --- Tracer les courbes d'apprentissage ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs_x = range(1, len(history["train_loss"]) + 1)

# Graphique 1 : Loss
ax1.plot(epochs_x, history["train_loss"], label="Train", color='steelblue')
ax1.plot(epochs_x, history["val_loss"], label="Validation", color='tomato')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss -- CNN simple")
ax1.legend()
ax1.grid(alpha=0.3)

# Graphique 2 : Accuracy
ax2.plot(epochs_x, history["train_acc"], label="Train", color='steelblue')
ax2.plot(epochs_x, history["val_acc"], label="Validation", color='tomato')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy -- CNN simple")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("courbes_cnn_simple.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : courbes_cnn_simple.png")

# ==================================================
# Partie 6 : Normalisation
# ==================================================

def calculer_mean_std(dataset):
    """Calcule la moyenne et l'ecart-type par canal sur le dataset."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_batches = 0

    for images, _ in loader:
        # images : [batch, 3, H, W]
        # On moyenne sur les dimensions batch, H, W (dim 0, 2, 3)
        mean += images.mean(dim=[0, 2, 3])
        std += images.std(dim=[0, 2, 3])
        n_batches += 1

    mean /= n_batches
    std /= n_batches
    print(f"\nMean par canal (R, G, B) : {mean}")
    print(f"Std par canal (R, G, B) : {std}")
    return mean.tolist(), std.tolist()


# Dataset temporaire avec ToTensor seulement, pour calculer les stats
dataset_stats = MelonaDataset("melanoma-cancer-dataset/train", transform=transforms.ToTensor())
MEAN, STD = calculer_mean_std(dataset_stats)

# --- Visualiser l'effet de la normalisation ---
from transforms import transform_normalise

# Charger la même image avec et sans normalisation
train_dataset_norm = MelonaDataset("melanoma-cancer-dataset/train", transform=transform_normalise)

img_brut, _ = train_dataset[0]      # forme [3, H, W], valeurs entre 0 et 1
img_norm, _ = train_dataset_norm[0]  # forme [3, H, W], valeurs centrees

# Pour afficher l'image normalisée, il faut remettre les valeurs entre 0 et 1
# Formule : (x - min) / (max - min)
img_norm_display = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

# Convertir pour matplotlib : [C, H, W] -> [H, W, C]
img_brut_np = img_brut.permute(1, 2, 0).numpy()
img_norm_np = img_norm_display.permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_brut_np)
axes[0].set_title(f"Avant normalisation\nmin={img_brut.min():.3f}, max={img_brut.max():.3f}")
axes[0].axis("off")

axes[1].imshow(img_norm_np)
axes[1].set_title(f"Apres normalisation\nmin={img_norm.min():.3f}, max={img_norm.max():.3f}")
axes[1].axis("off")

plt.suptitle("Effet de la normalisation")
plt.tight_layout()
plt.savefig("normalisation_effet.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : normalisation_effet.png")

# --- Reentrainer avec normalisation ---
val_dataset_norm = MelonaDataset("melanoma-cancer-dataset/test", transform=transform_normalise)

train_loader_norm = DataLoader(train_dataset_norm, batch_size=32, shuffle=True, num_workers=4)
val_loader_norm = DataLoader(val_dataset_norm, batch_size=32, shuffle=False, num_workers=4)

# Nouveau modele (on repart de zero pour comparaison equitable)
model_norm = SimpleCNN(num_classes=num_classes).to(device)
optimizer_norm = optim.Adam(model_norm.parameters(), lr=1e-3)

history_norm = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

LANCER_TRAIN_PARTIE_6 = False

if LANCER_TRAIN_PARTIE_6:
    print("\n=== Entrainement avec normalisation ===\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model_norm, train_loader_norm, criterion, optimizer_norm, device)
        val_loss, val_acc = evaluate(model_norm, val_loader_norm, criterion, device)
        duree = time.time() - t0
        history_norm["train_loss"].append(train_loss)
        history_norm["val_loss"].append(val_loss)
        history_norm["train_acc"].append(train_acc)
        history_norm["val_acc"].append(val_acc)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")
    print("\n=== Entrainement avec normalisation termine ===")
else:
    history_norm["train_loss"] = [0.3980, 0.3356, 0.3030, 0.2765, 0.2459, 0.2203, 0.1850, 0.1636, 0.1316, 0.1074, 0.1138, 0.0887, 0.0737, 0.0621, 0.0575, 0.0552, 0.0591, 0.0455, 0.0354, 0.0340]
    history_norm["val_loss"]   = [0.3546, 0.2853, 0.2832, 0.2571, 0.2622, 0.2270, 0.2241, 0.2050, 0.1969, 0.2172, 0.2164, 0.2370, 0.2458, 0.2690, 0.4756, 0.2859, 0.4085, 0.3698, 0.3250, 0.2769]
    history_norm["train_acc"]  = [0.824, 0.856, 0.873, 0.884, 0.898, 0.911, 0.924, 0.935, 0.949, 0.958, 0.956, 0.966, 0.975, 0.977, 0.980, 0.981, 0.981, 0.986, 0.988, 0.989]
    history_norm["val_acc"]    = [0.865, 0.896, 0.894, 0.907, 0.919, 0.922, 0.921, 0.935, 0.940, 0.931, 0.935, 0.938, 0.935, 0.945, 0.922, 0.940, 0.933, 0.936, 0.941, 0.944]
    print("\n=== Train Partie 6 SKIPPE ===")

# --- Comparer les courbes : avec vs sans normalisation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_x = range(1, NUM_EPOCHS + 1)

# Loss
ax1.plot(epochs_x, history["val_loss"], label="Sans normalisation", color='steelblue')
ax1.plot(epochs_x, history_norm["val_loss"], label="Avec normalisation", color='tomato')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Val Loss")
ax1.set_title("Loss val : avec vs sans normalisation")
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy
ax2.plot(epochs_x, history["val_acc"], label="Sans normalisation", color='steelblue')
ax2.plot(epochs_x, history_norm["val_acc"], label="Avec normalisation", color='tomato')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Accuracy")
ax2.set_title("Accuracy val : avec vs sans normalisation")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("comparaison_normalisation.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : comparaison_normalisation.png")

# ==================================================
# Partie 7 : Data Augmentation
# ==================================================
from transforms import train_transform_aug, val_transform

# --- Visualiser l'effet de l'augmentation ---
# Charger une image PIL directement
chemin_aug = os.path.join(TRAIN_DIR, "Benign", sorted(os.listdir(os.path.join(TRAIN_DIR, "Benign")))[0])
img_originale = Image.open(chemin_aug).convert("RGB")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

# Afficher l'originale en premier
axes[0].imshow(img_originale)
axes[0].set_title("Originale")
axes[0].axis("off")

# Appliquer 7 fois la transformation aleatoire
for i in range(1, 8):
    img_aug = train_transform_aug(img_originale)  # tenseur normalise
    # Remettre entre 0 et 1 pour affichage (formule min-max)
    img_aug_display = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min())
    img_aug_np = img_aug_display.permute(1, 2, 0).numpy()

    axes[i].imshow(img_aug_np)
    axes[i].set_title(f"Augmentation {i}")
    axes[i].axis("off")

plt.suptitle("Effet de la data augmentation (7 versions aleatoires)")
plt.tight_layout()
plt.savefig("augmentation_effet.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : augmentation_effet.png")

# --- Reentrainer avec augmentation ---
train_dataset_aug = MelonaDataset("melanoma-cancer-dataset/train", transform=train_transform_aug)
val_dataset_aug = MelonaDataset("melanoma-cancer-dataset/test", transform=val_transform)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True, num_workers=6)
val_loader_aug = DataLoader(val_dataset_aug, batch_size=32, shuffle=False, num_workers=6)

# Nouveau modele vierge
model_aug = SimpleCNN(num_classes=num_classes).to(device)
optimizer_aug = optim.Adam(model_aug.parameters(), lr=1e-3)

history_aug = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

LANCER_TRAIN_PARTIE_7 = False

if LANCER_TRAIN_PARTIE_7:
    print("\n=== Entrainement avec augmentation ===\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model_aug, train_loader_aug, criterion, optimizer_aug, device)
        val_loss, val_acc = evaluate(model_aug, val_loader_aug, criterion, device)
        duree = time.time() - t0
        history_aug["train_loss"].append(train_loss)
        history_aug["val_loss"].append(val_loss)
        history_aug["train_acc"].append(train_acc)
        history_aug["val_acc"].append(val_acc)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")
    print("\n=== Entrainement avec augmentation termine ===")
else:
    history_aug["train_loss"] = [0.4426, 0.3842, 0.3629, 0.3616, 0.3379, 0.3295, 0.3118, 0.3039, 0.2998, 0.2874, 0.2811, 0.2779, 0.2741, 0.2643, 0.2611, 0.2607, 0.2573, 0.2526, 0.2453, 0.2412]
    history_aug["val_loss"]   = [0.3905, 0.3638, 0.2985, 0.3041, 0.3411, 0.2769, 0.3128, 0.2789, 0.2797, 0.2660, 0.2776, 0.2782, 0.2536, 0.2393, 0.2583, 0.3030, 0.2551, 0.2252, 0.2505, 0.2381]
    history_aug["train_acc"]  = [0.802, 0.831, 0.839, 0.838, 0.854, 0.857, 0.869, 0.868, 0.873, 0.877, 0.884, 0.883, 0.883, 0.886, 0.890, 0.889, 0.891, 0.895, 0.898, 0.900]
    history_aug["val_acc"]    = [0.856, 0.834, 0.889, 0.879, 0.874, 0.895, 0.887, 0.897, 0.892, 0.915, 0.894, 0.903, 0.923, 0.913, 0.904, 0.877, 0.907, 0.925, 0.917, 0.925]
    print("\n=== Train Partie 7 SKIPPE ===")
# --- Comparaison des 3 entrainements ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_x = range(1, NUM_EPOCHS + 1)

# Loss val
ax1.plot(epochs_x, history["val_loss"],     label="Sans norm", color='steelblue')
ax1.plot(epochs_x, history_norm["val_loss"], label="Avec norm", color='tomato')
ax1.plot(epochs_x, history_aug["val_loss"],  label="Avec norm + aug", color='green')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss")
ax1.set_title("Loss val : comparaison des pipelines"); ax1.legend(); ax1.grid(alpha=0.3)

# Accuracy val
ax2.plot(epochs_x, history["val_acc"],     label="Sans norm", color='steelblue')
ax2.plot(epochs_x, history_norm["val_acc"], label="Avec norm", color='tomato')
ax2.plot(epochs_x, history_aug["val_acc"],  label="Avec norm + aug", color='green')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy")
ax2.set_title("Accuracy val : comparaison des pipelines"); ax2.legend(); ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("comparaison_augmentation.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : comparaison_augmentation.png")

# ==================================================
# Partie 8 : Comparer les optimiseurs et le learning rate
# ==================================================

# Mettre False pour skipper cette partie si deja fait
LANCER_PARTIE_8 = False

if LANCER_PARTIE_8:
    # On reutilise les loaders avec augmentation et normalisation (les meilleurs)
    # train_loader_aug et val_loader_aug ont deja ete crees a la Partie 7

    configs = [
        {"nom": "Adam_lr1e-3", "type": "adam", "lr": 1e-3},
        {"nom": "Adam_lr1e-4", "type": "adam", "lr": 1e-4},
        {"nom": "SGD_lr1e-2_mom09", "type": "sgd",  "lr": 1e-2, "momentum": 0.9},
    ]

    histories_optim = {}  # dict pour stocker l'historique de chaque config

    for cfg in configs:
        print(f"\n=== Entrainement : {cfg['nom']} ===\n")

        # Nouveau modele vierge pour chaque config
        model_optim = SimpleCNN(num_classes=num_classes).to(device)

        # Choix de l'optimiseur selon la config
        if cfg["type"] == "adam":
            optimizer_optim = optim.Adam(model_optim.parameters(), lr=cfg["lr"])
        elif cfg["type"] == "sgd":
            optimizer_optim = optim.SGD(
                model_optim.parameters(),
                lr=cfg["lr"],
                momentum=cfg["momentum"]
            )

        history_cfg = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model_optim, train_loader_aug, criterion, optimizer_optim, device)
            val_loss, val_acc = evaluate(model_optim, val_loader_aug, criterion, device)
            duree = time.time() - t0

            history_cfg["train_loss"].append(train_loss)
            history_cfg["val_loss"].append(val_loss)
            history_cfg["train_acc"].append(train_acc)
            history_cfg["val_acc"].append(val_acc)

            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
                  f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
                  f"{duree:.1f}s")

        histories_optim[cfg["nom"]] = history_cfg
        print(f"=== Fin {cfg['nom']} ===")

    # --- Comparaison des val_acc des 3 configs ---
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_x = range(1, NUM_EPOCHS + 1)

    couleurs = {'Adam_lr1e-3': 'steelblue', 'Adam_lr1e-4': 'tomato', 'SGD_lr1e-2_mom09': 'green'}

    for nom, h in histories_optim.items():
        ax.plot(epochs_x, h["val_acc"], label=nom, color=couleurs[nom])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Comparaison des optimiseurs et learning rates")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig("comparaison_optimiseurs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n>>> Figure sauvegardee : comparaison_optimiseurs.png")

    # --- Tableau recapitulatif des meilleures accuracies val ---
    print("\n=== Tableau recapitulatif des meilleures accuracies val ===")
    print(f"{'Configuration':<22} | {'Best val_acc':<12} | {'Epoch':<6}")
    print("-" * 50)
    for nom, h in histories_optim.items():
        best_acc = max(h["val_acc"])
        best_epoch = h["val_acc"].index(best_acc) + 1
        print(f"{nom:<22} | {best_acc:.4f}      | {best_epoch}")


        # ==================================================
# Partie 9 : Learning Rate Scheduler
# ==================================================

LANCER_PARTIE_9 = False
history_sched = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if LANCER_PARTIE_9:
    print("\n=== Entrainement avec scheduler StepLR ===\n")

    # Nouveau modele vierge, optimiseur Adam lr=1e-3 (meme que Partie 5-7)
    model_sched = SimpleCNN(num_classes=num_classes).to(device)
    optimizer_sched = optim.Adam(model_sched.parameters(), lr=1e-3)

    # Scheduler : divise le lr par 10 toutes les 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_sched, step_size=7, gamma=0.1)

    history_sched = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    lrs = []  # historique des learning rates pour le graphique

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model_sched, train_loader_aug, criterion, optimizer_sched, device)
        val_loss, val_acc = evaluate(model_sched, val_loader_aug, criterion, device)
        duree = time.time() - t0

        # Recuperer le lr courant AVANT le step
        lrs.append(optimizer_sched.param_groups[0]['lr'])
        scheduler.step()  # IMPORTANT : appele apres chaque epoch

        history_sched["train_loss"].append(train_loss)
        history_sched["val_loss"].append(val_loss)
        history_sched["train_acc"].append(train_acc)
        history_sched["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
              f"LR {lrs[-1]:.2e} | {duree:.1f}s")


    print("\n=== Entrainement avec scheduler termine ===")

    # --- Graphique 1 : Comparaison accuracy avec/sans scheduler ---
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_x = range(1, NUM_EPOCHS + 1)

    # Sans scheduler : on reutilise history_aug (Adam lr=1e-3 + norm + aug, Partie 7)
    ax.plot(epochs_x, history_aug["val_acc"],   label="Sans scheduler (lr=1e-3 fixe)", color='steelblue')
    ax.plot(epochs_x, history_sched["val_acc"], label="Avec scheduler StepLR", color='tomato')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Comparaison : avec vs sans scheduler")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig("comparaison_scheduler.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(">>> Figure sauvegardee : comparaison_scheduler.png")

    # --- Graphique 2 : Evolution du learning rate (echelle log) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_x, lrs, marker='o', color='purple')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Evolution du learning rate (StepLR : step_size=7, gamma=0.1)")
    ax.set_yscale('log')  # echelle logarithmique
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("evolution_lr.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(">>> Figure sauvegardee : evolution_lr.png")

    # Affichage du best score
    best_acc_sched = max(history_sched["val_acc"])
    best_epoch_sched = history_sched["val_acc"].index(best_acc_sched) + 1
    print(f"\nMeilleure accuracy val avec scheduler : {best_acc_sched:.4f} (epoch {best_epoch_sched})")
else:
    history_sched["train_loss"] = [0.4543, 0.3841, 0.3632, 0.3386, 0.3266, 0.3097, 0.3005, 0.2678, 0.2610, 0.2574, 0.2555, 0.2464, 0.2476, 0.2421, 0.2397, 0.2427, 0.2400, 0.2383, 0.2400, 0.2398]
    history_sched["val_loss"]   = [0.3778, 0.4175, 0.2968, 0.2914, 0.2974, 0.2850, 0.2840, 0.2532, 0.2471, 0.2587, 0.2345, 0.2426, 0.2426, 0.2295, 0.2314, 0.2337, 0.2330, 0.2303, 0.2299, 0.2295]
    history_sched["train_acc"]  = [0.794, 0.826, 0.839, 0.855, 0.862, 0.870, 0.873, 0.884, 0.891, 0.892, 0.892, 0.897, 0.894, 0.899, 0.900, 0.896, 0.899, 0.898, 0.898, 0.900]
    history_sched["val_acc"]    = [0.834, 0.809, 0.881, 0.889, 0.885, 0.900, 0.905, 0.914, 0.915, 0.907, 0.916, 0.918, 0.918, 0.925, 0.921, 0.922, 0.922, 0.922, 0.923, 0.921]
    print("\n=== Train Partie 9 SKIPPE ===")
 # ==================================================
# Partie 11 : Transfer Learning avec ResNet18
# ==================================================
from model import charger_resnet18_gele

LANCER_PARTIE_11 = False

history_resnet = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if LANCER_PARTIE_11:
    print("\n=== Entrainement ResNet18 (backbone gele) ===\n")
    
    resnet = charger_resnet18_gele(num_classes).to(device)
    print("\n--- Parametres ResNet18 ---")
    compter_parametres(resnet)
    
    optimizer_resnet = optim.Adam(resnet.fc.parameters(), lr=1e-3)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(resnet, train_loader_aug, criterion, optimizer_resnet, device)
        val_loss, val_acc = evaluate(resnet, val_loader_aug, criterion, device)
        duree = time.time() - t0
        
        history_resnet["train_loss"].append(train_loss)
        history_resnet["val_loss"].append(val_loss)
        history_resnet["train_acc"].append(train_acc)
        history_resnet["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
              f"{duree:.1f}s")
    
    print("\n=== Entrainement ResNet termine ===")
else:
    history_resnet["train_loss"] = [0.4112, 0.3446, 0.3361, 0.3314, 0.3245, 0.3283, 0.3260, 0.3207, 0.3204, 0.3158, 0.3169, 0.3178, 0.3227, 0.3228, 0.3169, 0.3302, 0.3187, 0.3221, 0.3061, 0.3188]
    history_resnet["val_loss"]   = [0.3270, 0.3156, 0.3038, 0.3153, 0.2930, 0.2919, 0.2885, 0.2940, 0.2871, 0.3481, 0.3186, 0.2853, 0.2856, 0.2850, 0.3002, 0.2815, 0.2941, 0.2820, 0.2907, 0.2843]
    history_resnet["train_acc"]  = [0.812, 0.851, 0.854, 0.856, 0.859, 0.861, 0.858, 0.863, 0.861, 0.863, 0.864, 0.859, 0.860, 0.863, 0.862, 0.859, 0.863, 0.861, 0.869, 0.861]
    history_resnet["val_acc"]    = [0.868, 0.867, 0.874, 0.867, 0.880, 0.873, 0.876, 0.878, 0.883, 0.840, 0.869, 0.884, 0.886, 0.886, 0.878, 0.883, 0.879, 0.885, 0.878, 0.880]
    print("\n=== Train Partie 11 SKIPPE ===")

# --- Courbes ResNet gele (toujours regenere) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_x = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs_x, history_resnet["train_loss"], label="Train", color='steelblue')
ax1.plot(epochs_x, history_resnet["val_loss"], label="Validation", color='tomato')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss -- ResNet18 gele"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs_x, history_resnet["train_acc"], label="Train", color='steelblue')
ax2.plot(epochs_x, history_resnet["val_acc"], label="Validation", color='tomato')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy -- ResNet18 gele"); ax2.legend(); ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("courbes_resnet_gele.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : courbes_resnet_gele.png")
# ==================================================
# Partie 12 : Fine-tuning partiel du ResNet18
# ==================================================
from model import charger_resnet18_finetune, charger_resnet18_finetune_layer3_4

# Mettre False pour skipper apres le premier run
LANCER_PARTIE_12 = False

history_ft    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
history_ft_l3 = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if LANCER_PARTIE_12:
    # ===========================
    # Variante 1 : layer4 + fc
    # ===========================
    print("\n=== Fine-tuning ResNet18 : layer4 + fc ===\n")
    
    resnet_ft = charger_resnet18_finetune(num_classes).to(device)
    print("\n--- Parametres ResNet18 fine-tune layer4+fc ---")
    compter_parametres(resnet_ft)
    
    optimizer_ft = optim.Adam([
        {"params": resnet_ft.layer4.parameters(), "lr": 1e-4},
        {"params": resnet_ft.fc.parameters(),     "lr": 1e-3},
    ])
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(resnet_ft, train_loader_aug, criterion, optimizer_ft, device)
        val_loss, val_acc = evaluate(resnet_ft, val_loader_aug, criterion, device)
        duree = time.time() - t0
        
        history_ft["train_loss"].append(train_loss)
        history_ft["val_loss"].append(val_loss)
        history_ft["train_acc"].append(train_acc)
        history_ft["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
              f"{duree:.1f}s")
    
    print("\n=== Fine-tuning layer4 termine ===")
    
    # ===========================
    # Variante 2 : layer3 + layer4 + fc
    # ===========================
    print("\n=== Fine-tuning ResNet18 : layer3 + layer4 + fc ===\n")
    
    resnet_ft2 = charger_resnet18_finetune_layer3_4(num_classes).to(device)
    print("\n--- Parametres ResNet18 fine-tune layer3+layer4+fc ---")
    compter_parametres(resnet_ft2)
    
    optimizer_ft2 = optim.Adam([
        {"params": resnet_ft2.layer3.parameters(), "lr": 1e-4},
        {"params": resnet_ft2.layer4.parameters(), "lr": 1e-4},
        {"params": resnet_ft2.fc.parameters(),     "lr": 1e-3},
    ])
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(resnet_ft2, train_loader_aug, criterion, optimizer_ft2, device)
        val_loss, val_acc = evaluate(resnet_ft2, val_loader_aug, criterion, device)
        duree = time.time() - t0
        
        history_ft_l3["train_loss"].append(train_loss)
        history_ft_l3["val_loss"].append(val_loss)
        history_ft_l3["train_acc"].append(train_acc)
        history_ft_l3["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
              f"{duree:.1f}s")
    
    print("\n=== Fine-tuning layer3+layer4 termine ===")
else:
    # Valeurs sauvegardees - layer4 (de ton entrainement precedent)
    history_ft["train_loss"] = [0.3024, 0.2423, 0.2156, 0.1815, 0.1658, 0.1469, 0.1399, 0.1172, 0.1053, 0.0964, 0.0812, 0.0863, 0.0670, 0.0744, 0.0653, 0.0595, 0.0541, 0.0570, 0.0473, 0.0463]
    history_ft["val_loss"]   = [0.1937, 0.1953, 0.2281, 0.2111, 0.2371, 0.1907, 0.2115, 0.2154, 0.2285, 0.1726, 0.1883, 0.2382, 0.2052, 0.1857, 0.1879, 0.2192, 0.2220, 0.2019, 0.1980, 0.2234]
    history_ft["train_acc"]  = [0.869, 0.897, 0.910, 0.927, 0.933, 0.941, 0.945, 0.954, 0.958, 0.963, 0.969, 0.969, 0.974, 0.973, 0.976, 0.978, 0.980, 0.980, 0.982, 0.984]
    history_ft["val_acc"]    = [0.925, 0.929, 0.896, 0.909, 0.890, 0.929, 0.918, 0.929, 0.926, 0.935, 0.936, 0.927, 0.941, 0.937, 0.941, 0.941, 0.926, 0.937, 0.942, 0.941]
    
    # Valeurs layer3+layer4 - placeholder, sera rempli apres premier run
    # (laisse vide pour l'instant, on rempliera apres)
    # Variante 2 : layer3 + layer4
    history_ft_l3["train_loss"] = [0.3042, 0.2356, 0.2033, 0.1769, 0.1586, 0.1415, 0.1313, 0.1038, 0.0930, 0.0878, 0.0869, 0.0590, 0.0660, 0.0525, 0.0526, 0.0451, 0.0473, 0.0514, 0.0393, 0.0402]
    history_ft_l3["val_loss"]   = [0.2230, 0.1880, 0.2095, 0.1999, 0.1572, 0.1810, 0.1668, 0.1354, 0.1991, 0.1341, 0.1357, 0.1714, 0.1767, 0.1662, 0.1731, 0.2106, 0.1803, 0.1711, 0.1628, 0.1894]
    history_ft_l3["train_acc"]  = [0.871, 0.900, 0.914, 0.930, 0.937, 0.943, 0.949, 0.959, 0.965, 0.967, 0.969, 0.978, 0.976, 0.980, 0.980, 0.985, 0.982, 0.981, 0.986, 0.986]
    history_ft_l3["val_acc"]    = [0.909, 0.935, 0.913, 0.908, 0.939, 0.933, 0.934, 0.954, 0.927, 0.947, 0.954, 0.946, 0.947, 0.949, 0.952, 0.947, 0.943, 0.949, 0.954, 0.947]
    
    print("\n=== Train Partie 12 SKIPPE ===")

# --- Comparaison : CNN maison / ResNet gele / ResNet FT layer4 / ResNet FT layer3+4 ---
fig, ax = plt.subplots(figsize=(12, 6))
epochs_x = range(1, NUM_EPOCHS + 1)

ax.plot(epochs_x, history_aug["val_acc"],    label="CNN maison (norm + aug)",      color='steelblue', linewidth=2)
ax.plot(epochs_x, history_resnet["val_acc"], label="ResNet18 gele",                color='gray',      linewidth=2)
ax.plot(epochs_x, history_ft["val_acc"],     label="ResNet18 FT layer4+fc",        color='tomato',    linewidth=2)
ax.plot(epochs_x, history_ft_l3["val_acc"],  label="ResNet18 FT layer3+layer4+fc", color='green',     linewidth=2)

ax.set_xlabel("Epoch")
ax.set_ylabel("Val Accuracy")
ax.set_title("Comparaison : CNN maison vs ResNet18 (gele vs fine-tune)")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.7, 1.0)

plt.tight_layout()
plt.savefig("comparaison_finetuning.png", dpi=150, bbox_inches='tight')
plt.close()
print(">>> Figure sauvegardee : comparaison_finetuning.png")

# --- Tableau recapitulatif final ---
print("\n=== Tableau recapitulatif final ===")
print(f"{'Modele':<40} | {'Best val_acc':<13}")
print("-" * 60)
print(f"{'CNN simple (Partie 5)':<40} | {max(history['val_acc']):.4f}")
print(f"{'CNN + norm (Partie 6)':<40} | {max(history_norm['val_acc']):.4f}")
print(f"{'CNN + norm + aug (Partie 7)':<40} | {max(history_aug['val_acc']):.4f}")
print(f"{'CNN + scheduler (Partie 9)':<40} | {max(history_sched['val_acc']):.4f}")
print(f"{'ResNet18 gele (Partie 11)':<40} | {max(history_resnet['val_acc']):.4f}")
print(f"{'ResNet18 FT layer4 (Partie 12)':<40} | {max(history_ft['val_acc']):.4f}")
print(f"{'ResNet18 FT layer3+layer4 (Partie 12)':<40} | {max(history_ft_l3['val_acc']):.4f}")

# ==================================================
# Partie 13 : Analyse des erreurs (matrice de confusion)
# ==================================================
from sklearn.metrics import confusion_matrix, classification_report

LANCER_PARTIE_13 = True

if LANCER_PARTIE_13:
    print("\n=== Partie 13 : reentrainement du meilleur modele pour analyse ===\n")
    
    # Reentrainer le ResNet FT layer4 (meilleur modele simple, ~15 min)
    resnet_best = charger_resnet18_finetune(num_classes).to(device)
    
    optimizer_best = optim.Adam([
        {"params": resnet_best.layer4.parameters(), "lr": 1e-4},
        {"params": resnet_best.fc.parameters(),     "lr": 1e-3},
    ])
    
    print("Reentrainement rapide pour avoir le modele en memoire...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(resnet_best, train_loader_aug, criterion, optimizer_best, device)
        val_loss, val_acc = evaluate(resnet_best, val_loader_aug, criterion, device)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Acc val {val_acc:.3f}")
    
    print("\n=== Reentrainement termine, debut analyse ===\n")
    
    # --- Matrice de confusion ---
    resnet_best.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in val_loader_aug:
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet_best(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de confusion :")
    print(cm)
    print("\nClassification report :")
    print(classification_report(y_true, y_pred, target_names=val_dataset_aug.classes))
    
    # --- Visualisation de la matrice ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(val_dataset_aug.classes)
    ax.set_yticklabels(val_dataset_aug.classes)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Vrai label")
    ax.set_title("Matrice de confusion - ResNet18 FT layer4")
    
    for i in range(num_classes):
        for j in range(num_classes):
            valeur = cm[i, j]
            couleur = "white" if valeur > cm.max() / 2 else "black"
            ax.text(j, i, str(valeur), ha="center", va="center", color=couleur, fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(">>> Figure sauvegardee : matrice_confusion.png")
    
    # --- Collecter 8 erreurs et 8 succes ---
    erreurs = []
    succes = []
    
    with torch.no_grad():
        for i in range(len(val_dataset_aug)):
            img_tensor, vrai_label = val_dataset_aug[i]
            img_input = img_tensor.unsqueeze(0).to(device)
            output = resnet_best(img_input)
            pred_label = output.argmax(dim=1).item()
            
            if pred_label != vrai_label and len(erreurs) < 8:
                erreurs.append((img_tensor, vrai_label, pred_label))
            elif pred_label == vrai_label and len(succes) < 8:
                succes.append((img_tensor, vrai_label, pred_label))
            
            if len(erreurs) >= 8 and len(succes) >= 8:
                break
    
    print(f"\nErreurs collectees : {len(erreurs)} | Succes : {len(succes)}")
    
    # --- Affichage des 8 erreurs ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, (img_tensor, vrai, predit) in enumerate(erreurs):
        img_display = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        axes[i].imshow(img_display.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Vrai: {val_dataset_aug.classes[vrai]}\nPredit: {val_dataset_aug.classes[predit]}", color='red')
        axes[i].axis("off")
    plt.suptitle("8 exemples MAL classes")
    plt.tight_layout()
    plt.savefig("erreurs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(">>> Figure sauvegardee : erreurs.png")
    
    # --- Affichage des 8 succes ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, (img_tensor, vrai, predit) in enumerate(succes):
        img_display = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        axes[i].imshow(img_display.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Vrai: {val_dataset_aug.classes[vrai]}\n(correct)", color='green')
        axes[i].axis("off")
    plt.suptitle("8 exemples BIEN classes")
    plt.tight_layout()
    plt.savefig("succes.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(">>> Figure sauvegardee : succes.png")

