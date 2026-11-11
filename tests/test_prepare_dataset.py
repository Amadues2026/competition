"""Regression tests for disjoint data splits and COCO-to-YOLO conversion."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_dataset", ROOT / "D2C2/train/prepare_dataset.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data with spaces"
        self.output = self.root / "output with spaces"
        (self.data / "annotations").mkdir(parents=True)
        (self.data / "images").mkdir()
        self.coco = {
            "images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 80}
                       for i in range(10)],
            "categories": [{"id": 4, "name": "red"}, {"id": 9, "name": "blue"}],
            "annotations": [
                {"image_id": 0, "category_id": 4, "bbox": [10, 20, 20, 40]},
                {"image_id": 1, "category_id": 9, "bbox": [0, 0, 100, 80]},
            ],
        }
        for image in self.coco["images"]:
            (self.data / "images" / image["file_name"]).touch()

    def prepare(self, **kwargs):
        (self.data / "annotations/annotations.json").write_text(json.dumps(self.coco), encoding="utf-8")
        return MODULE.prepare_dataset(self.data, self.output, **kwargs)

    def test_split_is_disjoint_complete_and_reproducible(self):
        config = self.prepare(seed=7)
        train = set((self.output / "train.txt").read_text().splitlines())
        val = set((self.output / "val.txt").read_text().splitlines())
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse(train & val)
        self.assertEqual(len(train | val), 10)
        first = (self.output / "train.txt").read_bytes()
        self.prepare(seed=7)
        self.assertEqual(first, (self.output / "train.txt").read_bytes())
        self.assertIn("names:\n  0: red\n  1: blue", config.read_text())

    def test_labels_normalize_coordinates_and_keep_background_images(self):
        self.prepare()
        self.assertEqual((self.data / "labels/0.txt").read_text(),
                         "0 0.200000 0.500000 0.200000 0.500000\n")
        self.assertEqual((self.data / "labels/1.txt").read_text(),
                         "1 0.500000 0.500000 1.000000 1.000000\n")
        self.assertEqual((self.data / "labels/9.txt").read_text(), "")

    def test_nested_image_names_keep_distinct_labels(self):
        self.coco["images"][0]["file_name"] = "sequence/0.jpg"
        (self.data / "images/sequence").mkdir()
        (self.data / "images/sequence/0.jpg").touch()
        self.prepare()
        self.assertTrue((self.data / "labels/sequence/0.txt").is_file())

    def test_rejects_invalid_boxes_before_writing_labels(self):
        for box in ([0, 0, -1, 2], [90, 0, 20, 2], [0, 0, float("nan"), 2]):
            with self.subTest(box=box):
                self.coco["annotations"][0]["bbox"] = box
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse((self.data / "labels").exists())

    def test_rejects_duplicate_ids_and_escaping_paths(self):
        original = copy.deepcopy(self.coco)
        self.coco["images"][1]["id"] = 0
        with self.assertRaises(ValueError):
            self.prepare()
        self.coco = original
        self.coco["images"][0]["file_name"] = "../outside.jpg"
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rejects_missing_images_and_unknown_image_references(self):
        self.coco["annotations"][0]["image_id"] = 999
        with self.assertRaises(ValueError):
            self.prepare()
        self.coco["annotations"][0]["image_id"] = 0
        (self.data / "images/0.jpg").unlink()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rejects_invalid_split_fraction_and_too_small_dataset(self):
        for fraction in (0, 1, -0.5):
            with self.assertRaises(ValueError):
                self.prepare(val_fraction=fraction)
        self.coco["images"] = self.coco["images"][:1]
        with self.assertRaises(ValueError):
            self.prepare()

    def test_prepare_cli_works_from_an_unrelated_directory(self):
        self.prepare()
        result = subprocess.run(
            [sys.executable, str(ROOT / "D2C2/train/train_yolov5.py"),
             "--data-dir", str(self.data), "--output-dir", str(self.output), "--prepare-only"],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
