"""Image pipelines. Augmentation is applied to the training set only."""
import torchvision.transforms as T

from melanoma_cnn.config import IMAGE_SIZE, MEAN, STD

BASE = T.Compose([T.ToTensor()])

NORMALIZED = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

TRAIN_AUGMENTED = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# Images that do not come from the dataset (scripts/predict.py) must be resized first.
PREDICT = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def to_displayable(image):
    """[C, H, W] tensor with any value range -> [H, W, C] numpy array in [0, 1] for imshow."""
    image = (image - image.min()) / (image.max() - image.min())
    return image.permute(1, 2, 0).numpy()
