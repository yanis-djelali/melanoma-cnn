"""Smoke tests of the PyTorch code on fake images (CPU, a few seconds). Skipped without torch.

    python -m unittest discover tests
"""
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from PIL import Image
except ImportError:  # the log tests still run on a machine without torch
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTest(unittest.TestCase):
    # Values printed during the May 2026 runs (results/raw_logs).
    EXPECTED_PARAMS = {
        "cnn_simple": (12_869_410, 12_869_410),
        "resnet_frozen": (1_026, 11_177_538),
        "resnet_ft_layer4": (8_394_754, 11_177_538),
        "resnet_ft_layer34": (10_494_466, 11_177_538),
    }

    def test_parameter_counts_match_the_logs(self):
        from melanoma_cnn.experiments import build_model
        from melanoma_cnn.models import count_parameters
        for name, expected in self.EXPECTED_PARAMS.items():
            model = build_model(name, 2, pretrained=False)
            self.assertEqual(count_parameters(model), expected, name)

    def test_output_shapes(self):
        from melanoma_cnn.experiments import build_model
        x = torch.zeros(4, 3, 224, 224)
        for name in ("cnn_simple", "resnet_ft_layer4"):
            self.assertEqual(tuple(build_model(name, 2, pretrained=False)(x).shape), (4, 2))

    def test_resnet_optimizer_learning_rates(self):
        from melanoma_cnn.experiments import build_model, build_optimizer
        model = build_model("resnet_ft_layer34", 2, pretrained=False)
        lrs = [group["lr"] for group in build_optimizer("resnet_ft_layer34", model).param_groups]
        self.assertEqual(lrs, [1e-4, 1e-4, 1e-3])

    def test_step_lr_schedule(self):
        from melanoma_cnn.experiments import build_model, build_optimizer, build_scheduler
        model = build_model("cnn_steplr", 2)
        optimizer = build_optimizer("cnn_steplr", model)
        scheduler = build_scheduler("cnn_steplr", optimizer)
        lrs = []
        for _ in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertEqual([round(lr, 8) for lr in lrs], [1e-3] * 7 + [1e-4] * 7 + [1e-5] * 6)


@unittest.skipIf(torch is None, "torch is not installed")
class PipelineTest(unittest.TestCase):
    """Dataset -> DataLoader -> training loop on 8 random 224x224 images."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        generator = torch.Generator().manual_seed(0)
        for name in ("Benign", "Malignant"):
            (root / name).mkdir()
            for i in range(4):
                pixels = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8,
                                       generator=generator)
                Image.fromarray(pixels.numpy()).save(root / name / f"{i}.jpg")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset(self):
        from melanoma_cnn import transforms as T
        from melanoma_cnn.data import MelanomaDataset
        dataset = MelanomaDataset(self.root, T.TRAIN_AUGMENTED)
        self.assertEqual(len(dataset), 8)
        self.assertEqual(dataset.class_to_idx, {"Benign": 0, "Malignant": 1})
        image, label = dataset[5]
        self.assertEqual((tuple(image.shape), label), ((3, 224, 224), 1))

    def test_fit_two_epochs(self):
        from melanoma_cnn import transforms as T
        from melanoma_cnn.data import make_loader
        from melanoma_cnn.engine import fit, predict
        from melanoma_cnn.experiments import build_model, build_optimizer
        torch.manual_seed(0)
        loader = make_loader(self.root, T.NORMALIZED, shuffle=True, batch_size=4, num_workers=0)
        model = build_model("cnn_norm", 2)
        history = fit(model, loader, loader, nn.CrossEntropyLoss(),
                      build_optimizer("cnn_norm", model), torch.device("cpu"), epochs=2)
        self.assertEqual({len(v) for v in history.values()}, {2})
        labels, predicted, probs = predict(model, loader, torch.device("cpu"))
        self.assertEqual(tuple(probs.shape), (8, 2))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(8)))


if __name__ == "__main__":
    unittest.main()
