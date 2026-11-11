"""Prepare disjoint YOLO train/validation lists from red/blue COCO annotations."""

import json
import math
import random
from pathlib import Path, PurePosixPath


def prepare_dataset(data_dir, output_dir, val_fraction=0.2, seed=42):
    """Write labels and a reproducible image-level split; return dataset.yaml.

    Images without recognized annotations receive empty labels. Correlated
    frames still require a sequence-level split for generalization evaluation.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    coco = json.loads((data_dir / "annotations/annotations.json").read_text(encoding="utf-8"))
    images = coco["images"]
    if len(images) < 2:
        raise ValueError("At least two images are required for a disjoint split")
    classes = {c["id"]: {"red": 0, "blue": 1}[c["name"]]
               for c in coco["categories"] if c["name"] in ("red", "blue")}
    if set(classes.values()) != {0, 1}:
        raise ValueError("COCO categories must include red and blue")
    by_id, paths, label_paths, records = {}, set(), set(), []
    for image in images:
        name = PurePosixPath(image["file_name"])
        if name.is_absolute() or ".." in name.parts or "\\" in image["file_name"]:
            raise ValueError("Image names must be relative POSIX paths inside images/")
        path = (data_dir / "images" / str(name)).resolve()
        if not path.is_relative_to(data_dir / "images") or not path.is_file():
            raise ValueError(f"Missing or out-of-tree image: {name}")
        label = data_dir / "labels" / str(name.with_suffix(".txt"))
        if image["id"] in by_id or path in paths or label in label_paths:
            raise ValueError("Duplicate image ID, image path, or label path")
        width, height = image["width"], image["height"]
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError(f"Invalid image dimensions: {name}")
        by_id[image["id"]] = (width, height, label, [])
        paths.add(path)
        label_paths.add(label)
        records.append(path)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in by_id:
            raise ValueError("Annotation references an unknown image ID")
        if annotation["category_id"] not in classes:
            continue
        width, height, _, lines = by_id[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0
                or x < 0 or y < 0 or x + w > width or y + h > height):
            raise ValueError("Bounding boxes must have positive area and fit inside the image")
        values = ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
        lines.append(str(classes[annotation["category_id"]]) + " "
                     + " ".join(f"{v:.6f}" for v in values))
    # Validate all input before writing any output.
    for _, _, label, lines in by_id.values():
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    records.sort()
    random.Random(seed).shuffle(records)
    val_count = max(1, min(len(records) - 1, round(len(records) * val_fraction)))
    for name, members in (("val", records[:val_count]), ("train", records[val_count:])):
        (output_dir / f"{name}.txt").write_text(
            "".join(p.as_posix() + "\n" for p in members), encoding="utf-8")
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {json.dumps(data_dir.as_posix())}\n"
        f"train: {json.dumps((output_dir / 'train.txt').as_posix())}\n"
        f"val: {json.dumps((output_dir / 'val.txt').as_posix())}\n"
        "names:\n  0: red\n  1: blue\n", encoding="utf-8")
    return yaml_path
