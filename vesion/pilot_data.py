"""Disjoint, class-balanced real-image crops for the Round 1 pilot."""

from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .evaluate import UNKNOWN, Annotation
from .round1_data import Round1TrainingTiles, read_tiles

PROXY_UNKNOWN = (7, 13, 24)


@dataclass(frozen=True)
class Crop:
    split: str
    tile_id: str
    left: int
    top: int
    annotations: tuple

    @property
    def image_id(self):
        return f"{self.split}_{self.tile_id}"


def _choose_crops(root, split, class_ids, size, quotas, seed, excluded_ids=()):
    dataset = Round1TrainingTiles(root, split)
    data = {}
    pools = defaultdict(list)
    for index, tile in enumerate(dataset.tiles):
        if tile.image_id in excluded_ids:
            continue
        labels = dataset.read_annotations(index)
        data[index] = labels
        for class_id in {label for label, _ in labels}:
            pools[class_id].append(index)

    rng = random.Random(seed)
    used = set()
    roles = []
    for role, quota in zip(("calibration", "validation", "training"), quotas, strict=True):
        crops = []
        for class_id in sorted(class_ids, key=lambda value: (len(pools[value]), value)):
            if role == "training" and class_id in PROXY_UNKNOWN:
                continue
            remaining = [index for index in pools[class_id] if index not in used]
            for index in rng.sample(remaining, k=min(quota, len(remaining))):
                tile = dataset.tiles[index]
                matching = [box for label, box in data[index] if label == class_id]
                x1, y1, x2, y2 = rng.choice(matching)
                left = max(0, min(tile.width - size, int((x1 + x2 - size) / 2)))
                top = max(0, min(tile.height - size, int((y1 + y2 - size) / 2)))
                annotations = []
                for label, (x1, y1, x2, y2) in data[index]:
                    if role == "training" and label in PROXY_UNKNOWN:
                        continue
                    if not (left <= (x1+x2)/2 < left+size and top <= (y1+y2)/2 < top+size):
                        continue
                    coordinates = (max(x1 - left, 0), max(y1 - top, 0), min(x2 - left, size), min(y2 - top, size))
                    annotations.append((label, coordinates))
                crops.append(Crop(split, tile.image_id, left, top, tuple(annotations)))
                used.add(index)
        roles.append(crops)
    return roles


def make_pilot_split(root, size, base_shots, new_shots, eval_shots, seed):
    # The release reuses the same source tiles for A and B with different labels.
    # Reserve B source IDs before assigning any A training or evaluation crops.
    b_ids = {tile.image_id for tile in read_tiles(root / "dataset_B")}
    a = _choose_crops(root, "dataset_A", range(30), size, (eval_shots, eval_shots, base_shots), seed, b_ids)
    b = _choose_crops(root, "dataset_B", range(30, 40), size, (eval_shots, eval_shots, new_shots), seed + 1)
    return {split: dict(zip(("calibration", "validation", "training"), parts, strict=True))
            for split, parts in (("A", a), ("B", b))}


class PilotTiles(Dataset):
    def __init__(self, root, crops, size):
        self.root, self.crops, self.size = root, crops, size
        self.cache = {}

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        crop = self.crops[index]
        if crop not in self.cache:
            path = self.root / crop.split / "images" / f"{crop.tile_id}.tif"
            x, y = crop.left, crop.top
            with Image.open(path) as tile:
                image = tile.crop((x, y, x+self.size, y+self.size))
                self.cache[crop] = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return self.cache[crop], crop.annotations, crop.split == "dataset_A", crop


def collate(samples):
    images, annotations, background, records = zip(*samples)
    return torch.stack(images).float().div_(255), annotations, torch.tensor(background), records


def ground_truth(crops, class_map):
    return [Annotation(crop.image_id, UNKNOWN if class_id in PROXY_UNKNOWN else class_map[class_id], box)
            for crop in crops for class_id, box in crop.annotations]
