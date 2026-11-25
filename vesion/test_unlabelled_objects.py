"""Measure the effect of unlabelled training objects on proposal recall."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .detector import Detector, decide, propose, train
from .evaluate import IOU_THRESHOLD, evaluate, iou
from .shapes import BASE_CLASSES, add_unlabelled_diamonds, annotations, make_split

TRAIN_IMAGES = 900
TEST_IMAGES = 600
TRAINING_SEED = 17
TEST_SEED = 217
PRIMARY_THRESHOLD = 0.4
THRESHOLDS = (0.1, 0.25, PRIMARY_THRESHOLD, 0.55, 0.7, 0.85)
PROPOSAL_BUDGETS = (1, 3, 5, 10)


def _proposal_metrics(candidates, test) -> dict:
    by_image = defaultdict(list)
    for candidate in candidates:
        by_image[candidate.image_id].append(candidate)

    matches_by_shape = defaultdict(list)
    for scene in test:
        boxes = sorted(by_image[scene.image_id], key=lambda row: -row.objectness)
        for shape, truth_box in scene.objects:
            first_match = next(((rank, row.objectness) for rank, row in enumerate(boxes, 1)
                                if iou(row.box, truth_box) >= IOU_THRESHOLD), (float("inf"), -1.0))
            matches_by_shape[shape].append(first_match)

    report = {}
    for shape, matches in sorted(matches_by_shape.items()):
        report[shape] = {
            "objects": len(matches),
            "matched_at_objectness": {str(threshold): sum(score >= threshold for _, score in matches)
                                      for threshold in THRESHOLDS},
            "matched_at_proposal_budget": {str(budget): sum(rank <= budget for rank, _ in matches)
                                           for budget in PROPOSAL_BUDGETS},
        }
    return report


def _run(training, test, seed, epochs):
    torch.manual_seed(seed)
    model = Detector()
    train(model, training, len(BASE_CLASSES), epochs, 0.003, seed)
    candidates = propose(model, test, BASE_CLASSES, min_objectness=0.0)
    truth = annotations(test, BASE_CLASSES)
    predictions = decide(candidates, PRIMARY_THRESHOLD, 0.0)
    return {
        "shapes": _proposal_metrics(candidates, test),
        "known_only_mAP_at_primary_threshold": evaluate(predictions, truth, BASE_CLASSES)["mAP_K"],
        "proposals_per_tile_at_primary_threshold": len(predictions) / len(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/unlabelled_objects"))
    parser.add_argument("--seed", type=int, default=TRAINING_SEED, help="Weight and batch-order seed for both runs")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--overlay-fraction", type=float, default=0.30)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1 or not 1 / TRAIN_IMAGES <= args.overlay_fraction <= 1:
        parser.error("use positive epochs and threads and an overlay fraction covering at least one training tile")
    torch.set_num_threads(args.threads)

    clean = make_split(TRAINING_SEED, "base_train", TRAIN_IMAGES)
    overlay_count = round(len(clean) * args.overlay_fraction)
    treatment = add_unlabelled_diamonds(clean, TRAINING_SEED, overlay_count)
    test = make_split(TEST_SEED, "test", TEST_IMAGES)
    report = {
        "training_scene_seed": TRAINING_SEED,
        "model_seed": args.seed,
        "test_seed": TEST_SEED,
        "training_images_per_run": len(clean),
        "training_images_with_unlabelled_diamonds": overlay_count,
        "test_images": len(test),
        "epochs_per_run": args.epochs,
        "primary_objectness_threshold": PRIMARY_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "runs": {},
    }
    for condition, scenes in (("control", clean), ("unlabelled_diamonds", treatment)):
        print(f"Training {condition}", flush=True)
        report["runs"][condition] = _run(scenes, test, args.seed, args.epochs)

    for condition, run in report["runs"].items():
        diamond = run["shapes"]["diamond"]
        matched = diamond["matched_at_objectness"][str(PRIMARY_THRESHOLD)]
        print(f"{condition}: found {matched}/{diamond['objects']} diamonds "
              f"at objectness >= {PRIMARY_THRESHOLD:.2f}")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
