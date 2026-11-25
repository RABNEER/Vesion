"""Test whether low-confidence proposals improve unknown detection."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .compare_prototypes import _settings
from .detector import Candidate, Detector, decide, propose, train
from .evaluate import IOU_THRESHOLD, UNKNOWN, Prediction, evaluate, iou, read_predictions, write_predictions, write_truth
from .prototype_gate import decide_by_prototype, prototype_distances
from .shapes import BASE_CLASSES, add_unlabelled_diamonds, annotations, make_split
from .synthetic import _load
from .test_unlabelled_objects import PRIMARY_THRESHOLD, TRAIN_IMAGES, TRAINING_SEED

TEST_SEED = 317
TEST_IMAGES = 600
EXTRA_LIMIT = 5


def _training_scenes():
    clean = make_split(TRAINING_SEED, "base_train", TRAIN_IMAGES)
    return add_unlabelled_diamonds(clean, TRAINING_SEED, round(0.3 * len(clean)))


def _train_checkpoint(path: Path, scenes, seed: int) -> None:
    torch.manual_seed(seed)
    model = Detector()
    train(model, scenes, len(BASE_CLASSES), 14, 0.003, seed)
    torch.save({"model": model.state_dict(), "classes": list(BASE_CLASSES), "model_seed": seed}, path)


def _extra_candidates(candidates: list[Candidate]) -> list[Candidate]:
    counts = defaultdict(int)
    kept = []
    for candidate in sorted(candidates, key=lambda row: -row.objectness):
        image_id = candidate.image_id
        if candidate.objectness < PRIMARY_THRESHOLD and counts[image_id] < EXTRA_LIMIT:
            counts[image_id] += 1
            kept.append(candidate)
    return kept


def _unknown_additions(candidates, distances, floor, cutoff) -> list[Prediction]:
    predictions = []
    for candidate, distance in zip(candidates, distances, strict=True):
        if candidate.objectness >= floor and distance > cutoff:
            score = candidate.objectness * (distance / (distance + 1))
            predictions.append(Prediction(candidate.image_id, UNKNOWN, candidate.box, score))
    return predictions


def _calibrate_extras(normal, extras, distances, truth, normal_cutoff):
    cutoffs = {normal_cutoff}
    if distances:
        cutoffs.update(value for value in np.quantile(distances, np.linspace(0.0, 1.0, 40)).tolist()
                       if value > normal_cutoff)
    best_key = (-1.0, -1.0, -1.0)
    for floor in (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30):
        for cutoff in sorted(cutoffs):
            added = _unknown_additions(extras, distances, floor, cutoff)
            scores = evaluate(normal + added, truth, BASE_CLASSES)
            key = (scores["OE-Score"], scores["U-Prec"], -len(added))
            if key > best_key:
                best_key = key
                best = {"min_extra_objectness": floor, "min_extra_distance": cutoff,
                        "calibration": scores, "extra_flags_on_calibration": len(added)}
    return best


def _shape_hits(predictions, scenes) -> dict:
    unknown_boxes = defaultdict(list)
    for row in predictions:
        if row.class_name == UNKNOWN:
            unknown_boxes[row.image_id].append(row.box)

    counts = defaultdict(lambda: {"matched": 0, "objects": 0})
    for scene in scenes:
        for shape, truth_box in scene.objects:
            if shape not in BASE_CLASSES:
                counts[shape]["objects"] += 1
                counts[shape]["matched"] += any(iou(truth_box, box) >= IOU_THRESHOLD
                                               for box in unknown_boxes[scene.image_id])
    return dict(counts)


def _compare(checkpoint: Path, support, output: Path) -> None:
    model, saved = _load(checkpoint)
    if tuple(saved["classes"]) != BASE_CLASSES or "model_seed" not in saved:
        raise ValueError("use a checkpoint trained by this experiment")
    calibration = make_split(TRAINING_SEED, "calibration", 240)
    calibration_truth = annotations(calibration, BASE_CLASSES)
    candidates = propose(model, calibration, BASE_CLASSES, include_embeddings=True, min_objectness=0.0)
    known_only = evaluate(decide(candidates, PRIMARY_THRESHOLD, 0.0), calibration_truth, BASE_CLASSES)
    normal_settings = _settings(candidates, calibration_truth, BASE_CLASSES,
                                propose(model, support, BASE_CLASSES, include_embeddings=True),
                                annotations(support, BASE_CLASSES), PRIMARY_THRESHOLD, known_only["mAP_K"])
    _, prototypes, thresholds, baseline_calibration, shrinkage = normal_settings["prototype"]
    normal = decide_by_prototype(candidates, prototype_distances(candidates, prototypes), **thresholds)
    extras = _extra_candidates(candidates)
    distances = prototype_distances(extras, prototypes)
    extra_settings = _calibrate_extras(normal, extras, distances, calibration_truth, thresholds["max_distance"])
    improves_calibration = extra_settings["calibration"]["OE-Score"] > baseline_calibration["OE-Score"]

    test = make_split(TEST_SEED, "test", TEST_IMAGES)
    truth = annotations(test, BASE_CLASSES)
    candidates = propose(model, test, BASE_CLASSES, include_embeddings=True, min_objectness=0.0)
    baseline = decide_by_prototype(candidates, prototype_distances(candidates, prototypes), **thresholds)
    extras = _extra_candidates(candidates)
    added = _unknown_additions(extras, prototype_distances(extras, prototypes),
                               extra_settings["min_extra_objectness"], extra_settings["min_extra_distance"])

    write_truth(output / "truth.csv", truth)
    report = {
        "model_seed": saved["model_seed"], "training_tiles_with_unlabelled_diamonds": round(0.3 * len(support)),
        "test_seed": TEST_SEED, "test_images": len(test), "extra_box_limit_per_tile": EXTRA_LIMIT,
        "normal_thresholds": thresholds, "covariance_shrinkage": shrinkage,
        "extra_settings": extra_settings,
        "baseline_calibration": baseline_calibration, "extra_flags_on_test": len(added), "methods": {},
        "method_selected_on_calibration": "extra_boxes" if improves_calibration else "baseline",
    }
    for name, predictions in (("baseline", baseline), ("extra_boxes", baseline + added)):
        path = output / f"{name}.csv"
        write_predictions(path, predictions)
        scores = evaluate(read_predictions(path), truth, BASE_CLASSES)
        report["methods"][name] = {"test": scores, "unknown_shapes": _shape_hits(predictions, test)}
        print(f"{name}: OE-Score={scores['OE-Score']:.2f}, mAP_K={scores['mAP_K']:.3f}, "
              f"U-Rec={scores['U-Rec']:.3f}, U-Prec={scores['U-Prec']:.3f}, U-F1={scores['U-F1']:.3f}")

    path = output / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/extra_boxes"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint", type=Path, help="Reuse a detector checkpoint from this experiment")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    support = _training_scenes()
    checkpoint = args.checkpoint or args.output / "detector.pt"
    if args.checkpoint is None:
        print(f"Training with unlabelled diamonds, seed {args.seed}", flush=True)
        _train_checkpoint(checkpoint, support, args.seed)
    _compare(checkpoint, support, args.output)


if __name__ == "__main__":
    main()
