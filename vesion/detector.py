"""Small grid detector with separate box, objectness and class outputs."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evaluate import UNKNOWN, Box, Prediction, iou
from .shapes import CLASSES, IMAGE_SIZE, Scene

STRIDE = 8
GRID = IMAGE_SIZE // STRIDE
MAX_BOX_SIZE = 32
MIN_OBJECTNESS = 0.1


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for inputs, outputs, stride in ((1, 16, 2), (16, 24, 2), (24, 32, 2), (32, 48, 1)):
            layers.extend((nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
                           nn.BatchNorm2d(outputs), nn.SiLU()))
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(48, 5 + len(CLASSES), 1)

    def forward(self, images):
        return self.head(self.features(images)).permute(0, 2, 3, 1)


def _pixels(scenes: list[Scene]) -> torch.Tensor:
    return torch.from_numpy(np.stack([scene.pixels for scene in scenes]))[:, None].float() / 255


def _training_data(scenes: list[Scene]) -> tuple[torch.Tensor, ...]:
    boxes = torch.zeros((len(scenes), GRID, GRID, 4))
    labels = torch.full((len(scenes), GRID, GRID), -1, dtype=torch.long)
    for index, scene in enumerate(scenes):
        for name, (left, top, right, bottom) in scene.objects:
            cx, cy = (left + right) / (2 * STRIDE), (top + bottom) / (2 * STRIDE)
            x, y = int(cx), int(cy)
            if labels[index, y, x] != -1:
                raise ValueError(f"two objects in one grid cell in {scene.image_id}")
            labels[index, y, x] = CLASSES.index(name)
            boxes[index, y, x] = torch.tensor((cx - x, cy - y, (right - left) / MAX_BOX_SIZE,
                                               (bottom - top) / MAX_BOX_SIZE))
    return _pixels(scenes), boxes, labels


def train(model: Detector, scenes: list[Scene], class_count: int, epochs: int,
          learning_rate: float, seed: int) -> None:
    images, boxes, labels = _training_data(scenes)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for indices in torch.randperm(len(scenes), generator=generator).split(32):
            output = model(images[indices])
            batch_labels = labels[indices]
            foreground = batch_labels >= 0
            loss = F.binary_cross_entropy_with_logits(
                output[..., 0], foreground.float(), pos_weight=output.new_tensor(8.0)
            )
            if foreground.any():
                positive = output[foreground]
                loss = loss + F.cross_entropy(positive[:, 5:5 + class_count], batch_labels[foreground])
                loss = loss + 10 * F.smooth_l1_loss(positive[:, 1:5].sigmoid(), boxes[indices][foreground])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(f"  epoch {epoch + 1:02}/{epochs}: loss={total_loss / len(scenes):.4f}", flush=True)


@dataclass(frozen=True)
class Candidate:
    image_id: str
    box: Box
    objectness: float
    class_name: str
    class_probability: float
    embedding: torch.Tensor | None = None


@torch.inference_mode()
def propose(model: Detector, scenes: list[Scene], classes: tuple[str, ...], *,
            include_embeddings: bool = False, min_objectness: float = MIN_OBJECTNESS) -> list[Candidate]:
    model.eval()
    candidates = []
    for start in range(0, len(scenes), 32):
        batch = scenes[start:start + 32]
        feature_maps = model.features(_pixels(batch))
        outputs = model.head(feature_maps).permute(0, 2, 3, 1)
        for scene, output, feature_map in zip(batch, outputs, feature_maps):
            objectness = output[..., 0].sigmoid()
            positions = (objectness >= min_objectness).nonzero()
            rows, cols = positions.unbind(dim=1)
            scores = objectness[rows, cols].tolist()
            scores_order = sorted(range(len(scores)), key=lambda idx: -scores[idx])
            logits = output[rows, cols]
            boxes = logits[:, 1:5].sigmoid().tolist()
            probabilities, names = logits[:, 5:5 + len(classes)].softmax(dim=1).max(dim=1)
            embeddings = feature_map[:, rows, cols].T if include_embeddings else None
            kept = []
            for idx in scores_order:
                x, y, width, height = boxes[idx]
                cx, cy = (cols[idx].item() + x) * STRIDE, (rows[idx].item() + y) * STRIDE
                half_width, half_height = width * MAX_BOX_SIZE / 2, height * MAX_BOX_SIZE / 2
                box = (max(0.0, cx - half_width), max(0.0, cy - half_height),
                       min(IMAGE_SIZE, cx + half_width), min(IMAGE_SIZE, cy + half_height))
                if any(iou(box, other) >= 0.5 for other in kept):
                    continue
                kept.append(box)
                candidates.append(Candidate(scene.image_id, box, scores[idx], classes[names[idx].item()],
                                            probabilities[idx].item(),
                                            embeddings[idx] if embeddings is not None else None))
    return candidates


def decide(candidates: list[Candidate], min_objectness: float,
           min_class_probability: float) -> list[Prediction]:
    predictions = []
    for candidate in candidates:
        if candidate.objectness < min_objectness:
            continue
        label, confidence = candidate.class_name, candidate.class_probability
        if confidence < min_class_probability:
            label, confidence = UNKNOWN, 1 - confidence
        predictions.append(Prediction(candidate.image_id, label, candidate.box, candidate.objectness * confidence))
    return predictions
