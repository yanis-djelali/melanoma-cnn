import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Effectue une epoch d'entrainement. Retourne (loss_moyenne, accuracy)."""
    model.train()  # mode entrainement (active Dropout)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        # 1. Envoyer les donnees sur GPU
        images = images.to(device)
        labels = labels.to(device)

        # 2. Remettre les gradients a zero
        optimizer.zero_grad()

        # 3. Passer les images dans le reseau (forward)
        outputs = model(images)

        # 4. Calculer la loss
        loss = criterion(outputs, labels)

        # 5. Backpropagation + mise a jour des poids
        loss.backward()
        optimizer.step()

        # Suivi des metriques
        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evalue le modele sur un loader. Retourne (loss_moyenne, accuracy)."""
    model.eval()  # mode evaluation (desactive Dropout)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # pas de calcul de gradient -> plus rapide
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy