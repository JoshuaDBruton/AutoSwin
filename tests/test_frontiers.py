import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision import transforms as T

from autoswin.dataloader import get_dataloaders
from autoswin.mccv import seed_fold
from autoswin.models import FOUNDATION_MODELS
from autoswin.multilabel_metrics import macro_auc


class FrontiersTests(unittest.TestCase):
    def test_checkpoint_identity(self):
        self.assertEqual(FOUNDATION_MODELS["biomedclip"]["model_name"],
                         "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    def test_macro_auc_excludes_undefined_class(self):
        truth = np.array([[0, 1, 0], [1, 0, 0]])
        probs = np.array([[.1, .2, .4], [.9, .8, .5]])
        self.assertEqual(macro_auc(truth, probs), .5)
        self.assertTrue(np.isnan(macro_auc(np.zeros((2, 1)), np.zeros((2, 1)))))

    def test_fixed_split_seeded_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            split = {"train": list(range(8)), "val": [8, 9], "test": [10, 11]}
            (path / "split.json").write_text(json.dumps(split))
            with h5py.File(path / "data.h5", "w") as data:
                data["images"] = np.zeros((12, 8, 8), dtype=np.uint8)
                data["labels"] = np.arange(12)

            def order(fold):
                self.assertEqual(seed_fold(fold), 2026 + fold)
                train, val, test = get_dataloaders(
                    str(path / "data.h5"), T.ToTensor(), labels_key="labels",
                    split_file=str(path / "split.json"), batch_size=2)
                self.assertEqual(list(val.dataset.indices), split["val"])
                self.assertEqual(list(test.dataset.indices), split["test"])
                return torch.cat([labels for _, labels in train]).tolist()

            orders = [order(fold) for fold in range(5)]
            self.assertEqual(orders, [order(fold) for fold in range(5)])
            self.assertNotEqual(orders[0], orders[1])


if __name__ == "__main__":
    unittest.main()
