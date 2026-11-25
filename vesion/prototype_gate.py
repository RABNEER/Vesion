"""Reject detections far from every known class in the detector's feature space."""

from collections import defaultdict

import torch
from torch.nn import functional as F

from .detector import Candidate
from .evaluate import UNKNOWN, Annotation, Prediction, iou


def _embeddings(candidates: list[Candidate]) -> torch.Tensor:
    return torch.stack([candidate.embedding for candidate in candidates])


def _training_embeddings(candidates: list[Candidate], truth: list[Annotation],
                         classes: tuple[str, ...]) -> list[torch.Tensor]:
    """Use one best-localized training proposal per labelled object."""
    by_image = defaultdict(list)
    for candidate in candidates:
        by_image[candidate.image_id].append(candidate)

    examples = defaultdict(list)
    for annotation in truth:
        if annotation.class_name not in classes:
            continue
        options = by_image[annotation.image_id]
        if options:
            best = max(options, key=lambda candidate: iou(candidate.box, annotation.box))
            if iou(best.box, annotation.box) >= 0.5:
                examples[annotation.class_name].append(best)

    for name in classes:
        if not examples[name]:
            raise ValueError(f"no correctly localized training proposals for {name!r}")
    return [_embeddings(examples[name]) for name in classes]


def fit_prototypes(candidates: list[Candidate], truth: list[Annotation], classes: tuple[str, ...],
                   shrinkage: float = 0.05) -> dict[str, torch.Tensor]:
    features = _training_embeddings(candidates, truth, classes)
    return fit_prototype_features(features, shrinkage)


def fit_prototype_features(features: list[torch.Tensor], shrinkage: float = 0.05) -> dict[str, torch.Tensor]:
    features = [F.normalize(rows, dim=1) for rows in features]
    centers = torch.stack([rows.mean(dim=0) for rows in features])
    residuals = torch.cat([rows - center for rows, center in zip(features, centers)])
    covariance = residuals.T @ residuals / len(residuals)
    # Pool within-class variation so the few-shot class does not have to
    # estimate its own covariance. Shrink toward an isotropic covariance.
    scale = covariance.diag().mean().clamp_min(1e-8)
    regularized = (1 - shrinkage) * covariance + shrinkage * scale * torch.eye(len(covariance))
    return {"centers": centers, "covariance_factor": torch.linalg.cholesky(regularized)}


def prototype_distances(candidates: list[Candidate], prototypes: dict[str, torch.Tensor]) -> list[float]:
    if not candidates:
        return []
    embeddings = F.normalize(_embeddings(candidates), dim=1)
    centers = prototypes["centers"]
    deltas = (embeddings[:, None, :] - centers).flatten(end_dim=1)
    whitened = torch.linalg.solve_triangular(prototypes["covariance_factor"], deltas.T, upper=False)
    distances = whitened.square().sum(dim=0).reshape(len(embeddings), len(centers))
    return distances.amin(dim=1).tolist()


def decide_by_prototype(candidates: list[Candidate], distances: list[float],
                        min_objectness: float, max_distance: float) -> list[Prediction]:
    predictions = []
    for candidate, distance in zip(candidates, distances, strict=True):
        if candidate.objectness < min_objectness:
            continue
        if distance > max_distance:
            label = UNKNOWN
            confidence = distance / (distance + 1)
        else:
            label, confidence = candidate.class_name, candidate.class_probability
        predictions.append(Prediction(candidate.image_id, label, candidate.box, candidate.objectness * confidence))
    return predictions
