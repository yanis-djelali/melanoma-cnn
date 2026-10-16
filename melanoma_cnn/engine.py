"""Training and evaluation loops."""
import time

import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training set. Returns (mean loss, accuracy)."""
    model.train()  # Dropout active, BatchNorm uses batch statistics
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # loss is averaged over the batch: multiply back so the last, smaller batch weighs right
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Mean loss and accuracy without updating the weights."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device):
    """Return (true labels, predicted labels, class probabilities) as CPU tensors."""
    model.eval()
    labels_all, probs_all = [], []
    for images, labels in loader:
        probs_all.append(torch.softmax(model(images.to(device)), dim=1).cpu())
        labels_all.append(labels)
    probs = torch.cat(probs_all)
    return torch.cat(labels_all), probs.argmax(dim=1), probs


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs,
        scheduler=None, writer=None):
    """Train for `epochs` epochs and return the history (one value per epoch and per key).

    The printed line has the same format as the original logs, so scripts/import_logs.py
    can read new runs too. `writer` is an optional TensorBoard SummaryWriter.
    """
    history = {key: [] for key in
               ("train_loss", "val_loss", "train_acc", "val_acc", "lr", "seconds")}
    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[-1]["lr"]  # last group = classification head
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()  # once per epoch, after the optimizer steps
        seconds = time.time() - start

        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc, lr, seconds)):
            history[key].append(value)
        if writer is not None:
            log_epoch(writer, epoch, train_loss, val_loss, train_acc, val_acc)
        print(f"Epoch {epoch:3d}/{epochs} | Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | LR {lr:.2e} | {seconds:.1f}s")
    return history


def log_epoch(writer, epoch, train_loss, val_loss, train_acc, val_acc):
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
