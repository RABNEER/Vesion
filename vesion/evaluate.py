"""Local interpretation of the published Anveshak scoring rules."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

UNKNOWN = "unknown"
PREDICTION_FIELDS = ("image_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax")
TRUTH_FIELDS = ("image_id", "class_name", "xmin", "ymin", "xmax", "ymax")
IOU_THRESHOLD = 0.5
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Annotation:
    image_id: str
    class_name: str
    box: Box


@dataclass(frozen=True)
class Prediction(Annotation):
    score: float


def iou(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    width = min(lx2, rx2) - max(lx1, rx1)
    height = min(ly2, ry2) - max(ly1, ry1)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union


def _read_rows(path: Path, fields: tuple[str, ...]):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, strict=True)
        if tuple(next(reader, [])) != fields:
            raise ValueError(f"{path}: expected header {','.join(fields)}")
        for values in reader:
            try:
                image_id, class_name = values[:2]
                if len(values) != len(fields) or not image_id or not class_name:
                    raise ValueError("invalid row")
                box = tuple(float(value) for value in values[-4:])
                xmin, ymin, xmax, ymax = box
                if not all(map(math.isfinite, box)) or not (0 <= xmin < xmax and 0 <= ymin < ymax):
                    raise ValueError("invalid box coordinates")
                if fields == PREDICTION_FIELDS:
                    score = float(values[2])
                    if not 0 <= score <= 1:
                        raise ValueError("score must be between 0 and 1")
                    yield Prediction(image_id, class_name, box, score)
                else:
                    yield Annotation(image_id, class_name, box)
            except ValueError as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc


def read_predictions(path: Path) -> list[Prediction]:
    rows = list(_read_rows(path, PREDICTION_FIELDS))
    if len(set(rows)) != len(rows):
        raise ValueError(f"{path}: duplicate prediction rows")
    if any(count > 1000 for count in Counter(row.image_id for row in rows).values()):
        raise ValueError(f"{path}: more than 1,000 predictions for one image")
    return rows


def read_truth(path: Path) -> list[Annotation]:
    return list(_read_rows(path, TRUTH_FIELDS))


def write_predictions(path: Path, rows: list[Prediction]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(PREDICTION_FIELDS)
        for row in rows:
            writer.writerow((row.image_id, row.class_name, row.score, *row.box))


def write_truth(path: Path, rows: list[Annotation]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(TRUTH_FIELDS)
        for row in rows:
            writer.writerow((row.image_id, row.class_name, *row.box))


def _resolve_conflicts(predictions: list[Prediction]) -> list[Prediction]:
    kept = []
    by_image_and_type = defaultdict(list)
    # The stable sort gives priority to the earlier CSV row when scores tie.
    for prediction in sorted(predictions, key=lambda row: -row.score):
        is_unknown = prediction.class_name == UNKNOWN
        opposing = by_image_and_type[prediction.image_id, not is_unknown]
        if any(iou(prediction.box, other) >= IOU_THRESHOLD for other in opposing):
            continue
        by_image_and_type[prediction.image_id, is_unknown].append(prediction.box)
        kept.append(prediction)
    return kept


def _all_point_ap(matches: list[bool], truth_count: int) -> float:
    true_positives = 0
    precision_at_match = []
    for rank, matched in enumerate(matches, 1):
        if matched:
            true_positives += 1
            precision_at_match.append(true_positives / rank)

    # Each true positive increases recall by 1 / truth_count. Integrate
    # the precision envelope over those increments.
    best_precision = total = 0.0
    for precision in reversed(precision_at_match):
        best_precision = max(best_precision, precision)
        total += best_precision
    return total / truth_count


def evaluate(predictions: list[Prediction], truth: list[Annotation], known_classes=None) -> dict:
    if known_classes is None:
        known_classes = {row.class_name for row in truth if row.class_name != UNKNOWN}
    known_classes = set(known_classes)
    if UNKNOWN in known_classes:
        raise ValueError("unknown cannot be a known class")
    labels = known_classes | {UNKNOWN}
    invalid = ({row.class_name for row in predictions} | {row.class_name for row in truth}) - labels
    if invalid:
        raise ValueError(f"labels missing from the class list: {', '.join(sorted(invalid))}")

    unmatched = defaultdict(list)
    truth_counts = Counter()
    for row in truth:
        unmatched[row.image_id, row.class_name].append(row.box)
        truth_counts[row.class_name] += 1

    matches = defaultdict(list)
    kept = _resolve_conflicts(predictions)
    for row in kept:
        candidates = unmatched[row.image_id, row.class_name]
        overlaps = [iou(row.box, box) for box in candidates]
        best_index = max(range(len(overlaps)), key=overlaps.__getitem__, default=None)
        matched = best_index is not None and overlaps[best_index] >= IOU_THRESHOLD
        if matched:
            candidates.pop(best_index)
        matches[row.class_name].append(matched)

    per_class = {
        name: _all_point_ap(matches[name], truth_counts[name])
        for name in sorted(known_classes) if truth_counts[name]
    }
    map_known = sum(per_class.values()) / len(per_class) if per_class else 0.0
    unknown_tp = sum(matches[UNKNOWN])
    unknown_predictions = len(matches[UNKNOWN])
    unknown_truth = truth_counts[UNKNOWN]
    recall = unknown_tp / unknown_truth if unknown_truth else 0.0
    precision = unknown_tp / unknown_predictions if unknown_predictions else 0.0
    f1 = _harmonic_mean(recall, precision)
    return {
        "mAP_K": map_known,
        "U-Rec": recall,
        "U-Prec": precision,
        "U-F1": f1,
        "OE-Score": 100 * _harmonic_mean(map_known, f1),
        "known_AP": per_class,
        "classes_without_truth": sorted(known_classes - per_class.keys()),
        "unknown_counts": {"matched": unknown_tp, "predictions": unknown_predictions, "truth": unknown_truth},
        "conflicting_predictions_removed": len(predictions) - len(kept),
    }


def _harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--known-classes", nargs="+", help="Known labels, including any absent from truth")
    args = parser.parse_args()
    try:
        scores = evaluate(read_predictions(args.predictions), read_truth(args.truth), args.known_classes)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
