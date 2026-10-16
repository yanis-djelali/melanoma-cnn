"""Dataset reading <root>/<class_name>/*.jpg and the matching DataLoader."""
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from melanoma_cnn.config import BATCH_SIZE, NUM_WORKERS


class MelanomaDataset(Dataset):
    """One sample = (transformed image, integer label). Labels follow the sorted class names."""

    def __init__(self, root, transform=None):
        root = Path(root)
        self.transform = transform
        self.classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        # Files are sorted so that the sample order does not depend on the file system.
        self.samples = [
            (path, self.class_to_idx[name])
            for name in self.classes
            for path in sorted((root / name).glob("*.jpg"))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_loader(root, transform, shuffle, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    dataset = MelanomaDataset(root, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
