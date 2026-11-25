"""Compare prototype rejection and softmax on identical synthetic detections."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .detector import decide, propose
from .evaluate import evaluate, write_predictions, write_truth
from .prototype_gate import decide_by_prototype, fit_prototypes, prototype_distances
from .shapes import BASE_CLASSES, annotations, make_split
from .synthetic import _load, _predict, _unknown_proposal_recall


def _settings(candidates, truth, classes, support_candidates, support_truth,
              min_objectness, baseline_map):
    kept = [row for row in candidates if row.objectness >= min_objectness]
    if not kept:
        raise ValueError("the checkpoint produced no usable calibration boxes")
    best = {}
    for shrinkage in (0.01, 0.05, 0.10):
        prototypes = fit_prototypes(support_candidates, support_truth, classes, shrinkage)
        distances = prototype_distances(kept, prototypes)
        cutoffs = np.quantile(distances, np.linspace(0.05, 0.99, 64))
        for cutoff in np.unique(cutoffs).tolist():
            thresholds = {"min_objectness": min_objectness, "max_distance": cutoff}
            predictions = decide_by_prototype(kept, distances, **thresholds)
            scores = evaluate(predictions, truth, classes)
            key = (scores["OE-Score"], scores["mAP_K"], scores["U-F1"])
            result = (key, prototypes, thresholds, scores, shrinkage)
            modes = ["prototype"]
            if scores["mAP_K"] >= baseline_map - 0.02:
                modes.append("prototype_balanced")
            for mode in modes:
                if mode not in best or key > best[mode][0]:
                    best[mode] = result
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Synthetic checkpoint with the softmax rule")
    parser.add_argument("--output", type=Path, default=Path("runs/prototype_comparison"))
    parser.add_argument("--test-seed", type=int, default=117)
    parser.add_argument("--test-images", type=int, default=600)
    parser.add_argument("--shots", type=int, default=12, help="Original number of few-shot training images")
    args = parser.parse_args()

    model, saved = _load(args.checkpoint)
    seed, classes = saved["seed"], tuple(saved["classes"])
    if "prototypes" in saved:
        parser.error("use the original softmax checkpoint for comparison")
    if args.test_seed == seed or min(args.test_images, args.shots) < 1:
        parser.error("use a fresh test seed and positive image counts")
    torch.set_num_threads(saved["threads"])
    min_objectness = saved["thresholds"]["min_objectness"]

    support = make_split(seed, "base_train", 900)
    if classes != BASE_CLASSES:
        support += make_split(seed, "few_shot", args.shots)
    calibration = make_split(seed, "calibration", 120)
    calibration_truth = annotations(calibration, classes)
    calibration_candidates = propose(model, calibration, classes, include_embeddings=True)
    baseline_calibration = evaluate(decide(calibration_candidates, **saved["thresholds"]), calibration_truth, classes)
    settings = _settings(calibration_candidates, calibration_truth, classes,
                         propose(model, support, classes, include_embeddings=True), annotations(support, classes),
                         min_objectness, baseline_calibration["mAP_K"])

    test = make_split(args.test_seed, "test", args.test_images)
    truth = annotations(test, classes)
    candidates = propose(model, test, classes, include_embeddings=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_truth(args.output / "truth.csv", truth)
    baseline = decide(candidates, **saved["thresholds"])
    write_predictions(args.output / "softmax.csv", baseline)

    report = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "test_seed": args.test_seed,
        "test_images": args.test_images,
        "prototype_support_shots": args.shots if classes != BASE_CLASSES else 0,
        "unknown_proposal_recall": _unknown_proposal_recall(candidates, truth, min_objectness),
        "balanced_max_calibration_map_drop": 0.02,
        "methods": {"softmax": {"calibration": baseline_calibration, "test": evaluate(baseline, truth, classes)}},
    }
    for mode, (_, prototypes, thresholds, calibration_scores, shrinkage) in settings.items():
        predictions = decide_by_prototype(candidates, prototype_distances(candidates, prototypes), **thresholds)
        if [(row.image_id, row.box) for row in predictions] != [(row.image_id, row.box) for row in baseline]:
            raise RuntimeError(f"{mode} changed the boxes used in the comparison")
        path = args.output / f"{mode}.csv"
        checkpoint = args.output / f"{mode}.pt"
        write_predictions(path, predictions)
        torch.save({**saved, "prototypes": prototypes, "thresholds": thresholds,
                    "test_seed": args.test_seed, "test_images": len(test)}, checkpoint)
        reproduced_path = args.output / f"{mode}_reproduced.csv"
        write_predictions(reproduced_path, _predict(checkpoint, test))
        if path.read_bytes() != reproduced_path.read_bytes():
            raise RuntimeError(f"could not reproduce {path} from its checkpoint")
        report["methods"][mode] = {
            "thresholds": thresholds, "covariance_shrinkage": shrinkage, "calibration": calibration_scores,
            "test": evaluate(predictions, truth, classes), "checkpoint_csv_reproduced": True,
        }

    path = args.output / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, method in report["methods"].items():
        scores = method["test"]
        print(f"{name}: OE-Score={scores['OE-Score']:.2f}, mAP_K={scores['mAP_K']:.3f}, U-F1={scores['U-F1']:.3f}")
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
