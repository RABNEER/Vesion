"""Read and audit the released Anveshak Round 1 training data."""

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .evaluate import UNKNOWN, Box

TRAINING_SPLITS = ("dataset_A", "dataset_B")


@dataclass(frozen=True)
class Tile:
    image_id: str
    width: int
    height: int


def read_class_map(root: Path) -> dict[int, str]:
    with (root / "class_list.csv").open(newline="", encoding="utf-8-sig") as file:
        return {int(row["class_id"]): row["label"] for row in csv.DictReader(file)}


def read_tiles(directory: Path) -> list[Tile]:
    with (directory / "tiles.csv").open(newline="", encoding="utf-8-sig") as file:
        return [Tile(row["tile_id"], int(row["width"]), int(row["height"])) for row in csv.DictReader(file)]


class Round1TrainingTiles(Dataset):
    """Load RGB images and variable-length targets in original tile coordinates."""

    def __init__(self, root: Path, split: str):
        if split not in TRAINING_SPLITS:
            raise ValueError(f"training split must be one of {TRAINING_SPLITS}")
        self.directory = root / split
        self.tiles = read_tiles(self.directory)
        self.class_map = read_class_map(root)

    def __len__(self) -> int:
        return len(self.tiles)

    def read_annotations(self, index: int) -> list[tuple[int, Box]]:
        tile = self.tiles[index]
        path = self.directory / "labels" / f"{tile.image_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["tile_id"], data["width"], data["height"], data["bbox_format"]) != (
            tile.image_id, tile.width, tile.height, "xyxy"
        ):
            raise ValueError(f"{path}: annotation metadata does not match the manifest")

        annotations = []
        for row in data["annotations"]:
            class_id = row["class_id"]
            left, top, right, bottom = row["bbox"]
            if self.class_map.get(class_id) != row["label"]:
                raise ValueError(f"{path}: class ID and label do not match the class list")
            if not (0 <= left < right <= tile.width and 0 <= top < bottom <= tile.height):
                raise ValueError(f"{path}: invalid box coordinates")
            annotations.append((class_id, (left, top, right, bottom)))
        return annotations

    def __getitem__(self, index: int):
        tile = self.tiles[index]
        path = self.directory / "images" / f"{tile.image_id}.tif"
        with Image.open(path) as image:
            if image.size != (tile.width, tile.height) or image.mode != "RGB":
                raise ValueError(f"{path}: expected an RGB image matching the manifest size")
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        annotations = self.read_annotations(index)
        target = {
            "image_id": tile.image_id,
            "boxes": torch.tensor([box for _, box in annotations], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([label for label, _ in annotations], dtype=torch.int64),
        }
        return pixels.float() / 255, target


def collate_detection_batch(samples):
    """Keep each tile's boxes and labels together, including empty targets."""
    return tuple(zip(*samples))


def _check_files(directory: Path, tiles: list[Tile], *, training: bool) -> None:
    ids = {tile.image_id for tile in tiles}
    if len(ids) != len(tiles):
        raise ValueError(f"{directory}: duplicate tile IDs")
    groups = [("images", ".tif")]
    if training:
        groups.append(("labels", ".json"))
    for name, suffix in groups:
        files = {path.stem for path in (directory / name).glob(f"*{suffix}") if path.is_file()}
        if files != ids:
            raise ValueError(f"{directory / name}: {len(ids - files)} missing and {len(files - ids)} extra files")


def audit(root: Path) -> dict:
    class_map = read_class_map(root)
    report = {"known_classes": sum(label != UNKNOWN for label in class_map.values()), "splits": {}}
    for split in (*TRAINING_SPLITS, "test"):
        training = split in TRAINING_SPLITS
        dataset = Round1TrainingTiles(root, split) if training else None
        tiles = dataset.tiles if dataset is not None else read_tiles(root / split)
        _check_files(root / split, tiles, training=training)
        sizes = Counter(f"{tile.width}x{tile.height}" for tile in tiles)
        result = {"tiles": len(tiles), "tile_sizes": dict(sizes)}
        if dataset is not None:
            counts = Counter()
            tiles_per_class = Counter()
            boxes_per_tile = []
            sizes = []
            for index in range(len(dataset)):
                annotations = dataset.read_annotations(index)
                labels = [label for label, _ in annotations]
                counts.update(labels)
                tiles_per_class.update(set(labels))
                boxes_per_tile.append(len(labels))
                sizes.extend((right - left, bottom - top) for _, (left, top, right, bottom) in annotations)
            result.update({
                "class_ids": sorted(counts), "boxes": sum(counts.values()),
                "boxes_by_class": dict(sorted(counts.items())),
                "tiles_by_class": dict(sorted(tiles_per_class.items())),
                "tiles_without_annotations": boxes_per_tile.count(0),
                "max_boxes_per_tile": max(boxes_per_tile, default=0),
                "median_box_width_height_px": np.median(sizes, axis=0).tolist() if sizes else None,
            })
        report["splits"][split] = result
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("round1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(audit(args.root), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
        print(f"Saved audit to {args.output}")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
