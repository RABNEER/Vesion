"""Run and reproduce Vesion's synthetic detect, doubt and adapt experiment."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from .detector import Detector, decide, propose, train
from .evaluate import Annotation, UNKNOWN, evaluate, iou, read_predictions, write_predictions, write_truth
from .prototype_gate import decide_by_prototype, prototype_distances
from .shapes import BASE_CLASSES, CLASSES, annotations, make_split

TEST_IMAGES = 200


def _calibrate(candidates, truth, classes):
    best_key = (-1.0, -1.0, -1.0)
    for objectness in (0.15, 0.25, 0.4, 0.55, 0.7, 0.85):
        for confidence in (0.0, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98, 0.995):
            scores = evaluate(decide(candidates, objectness, confidence), truth, classes)
            key = (scores["OE-Score"], scores["mAP_K"], scores["U-F1"])
            if key > best_key:
                best_key = key
                thresholds = {"min_objectness": objectness, "min_class_probability": confidence}
    return thresholds


def _load(checkpoint: Path):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Detector()
    model.load_state_dict(saved["model"])
    return model, saved


def _predict(checkpoint: Path, scenes):
    model, saved = _load(checkpoint)
    prototypes = saved.get("prototypes")
    candidates = propose(model, scenes, tuple(saved["classes"]), include_embeddings=prototypes is not None)
    if prototypes is not None:
        return decide_by_prototype(candidates, prototype_distances(candidates, prototypes), **saved["thresholds"])
    return decide(candidates, **saved["thresholds"])


def _unknown_proposal_recall(candidates, truth: list[Annotation], min_objectness: float) -> float:
    boxes = defaultdict(list)
    for candidate in candidates:
        if candidate.objectness >= min_objectness:
            boxes[candidate.image_id].append(candidate.box)
    unknown_objects = [row for row in truth if row.class_name == UNKNOWN]
    matched = sum(any(iou(row.box, box) >= 0.5 for box in boxes[row.image_id]) for row in unknown_objects)
    return matched / len(unknown_objects) if unknown_objects else 0.0


def _run_stage(name, model, classes, output, seed, calibration, test):
    calibration_candidates = propose(model, calibration, classes)
    thresholds = _calibrate(calibration_candidates, annotations(calibration, classes), classes)
    checkpoint = output / f"{name}.pt"
    torch.save({"model": model.state_dict(), "classes": list(classes), "thresholds": thresholds,
                "seed": seed, "test_images": len(test), "threads": torch.get_num_threads()}, checkpoint)

    candidates = propose(model, test, classes)
    truth = annotations(test, classes)
    truth_path = output / f"{name}_truth.csv"
    prediction_path = output / f"{name}_predictions.csv"
    closed_set_path = output / f"{name}_known_only.csv"
    write_truth(truth_path, truth)
    write_predictions(prediction_path, decide(candidates, **thresholds))
    write_predictions(closed_set_path, decide(candidates, thresholds["min_objectness"], 0.0))

    reproduced_path = output / f"{name}_reproduced.csv"
    write_predictions(reproduced_path, _predict(checkpoint, test))
    if prediction_path.read_bytes() != reproduced_path.read_bytes():
        raise RuntimeError(f"could not reproduce {prediction_path} from its checkpoint")

    return checkpoint, {
        "thresholds": thresholds,
        "unknown_proposal_recall": _unknown_proposal_recall(candidates, truth, thresholds["min_objectness"]),
        "known_only": evaluate(read_predictions(closed_set_path), truth, classes),
        "abstention": evaluate(read_predictions(prediction_path), truth, classes),
        "checkpoint_csv_reproduced": True,
    }


def run(args) -> None:
    torch.manual_seed(args.seed)
    training = make_split(args.seed, "base_train", 900)
    few_shot = make_split(args.seed, "few_shot", args.shots)
    calibration = make_split(args.seed, "calibration", 120)
    test = make_split(args.seed, "test", TEST_IMAGES)

    print("Stage A: train from random weights on circle and square", flush=True)
    model = Detector()
    train(model, training, len(BASE_CLASSES), args.epochs_a, 0.003, args.seed)
    checkpoint, stage_a = _run_stage("stage_a", model, BASE_CLASSES, args.output, args.seed, calibration, test)

    print(f"Stage B: resume Stage A with {args.shots} labelled triangles and 120 replay scenes", flush=True)
    model, _ = _load(checkpoint)
    replay = random.Random(args.seed).sample(training, 120)
    train(model, replay + few_shot * 5, len(CLASSES), args.epochs_b, 0.0015, args.seed + 1)
    _, stage_b = _run_stage("stage_b", model, CLASSES, args.output, args.seed, calibration, test)

    before, after = (stage["abstention"]["known_AP"] for stage in (stage_a, stage_b))
    old_map_before = sum(before[name] for name in BASE_CLASSES) / len(BASE_CLASSES)
    old_map_after = sum(after[name] for name in BASE_CLASSES) / len(BASE_CLASSES)
    metrics = {
        "seed": args.seed,
        "few_shot_images": args.shots,
        "stages": {"stage_a": stage_a, "stage_b": stage_b},
        "adaptation": {"old_mAP_before": old_map_before, "old_mAP_after": old_map_after,
                       "old_mAP_change": old_map_after - old_map_before, "new_AP": after["triangle"]},
    }
    report = args.output / "metrics.json"
    report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for name, stage in metrics["stages"].items():
        result = stage["abstention"]
        print(f"{name}: mAP_K={result['mAP_K']:.3f}, U-F1={result['U-F1']:.3f}, "
              f"OE-Score={result['OE-Score']:.2f}, unknown proposal recall={stage['unknown_proposal_recall']:.3f}")
    print(f"Old-class mAP: {old_map_before:.3f} -> {old_map_after:.3f}. Triangle AP: {after['triangle']:.3f}.")
    print(f"Saved results to {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs-a", type=int, default=14)
    parser.add_argument("--epochs-b", type=int, default=8)
    parser.add_argument("--shots", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, help="Only regenerate a predictions CSV from this checkpoint")
    args = parser.parse_args()
    if min(args.epochs_a, args.epochs_b, args.shots, args.threads) < 1:
        parser.error("epoch counts, shots and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        torch.set_num_threads(saved["threads"])
        test = make_split(saved.get("test_seed", saved["seed"]), "test", saved["test_images"])
        path = args.output / "predictions.csv"
        write_predictions(path, _predict(args.checkpoint, test))
        print(f"Saved {path}")
    else:
        torch.set_num_threads(args.threads)
        run(args)


if __name__ == "__main__":
    main()
