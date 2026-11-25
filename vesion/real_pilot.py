"""Run a from-scratch Set A → Set B pilot on released training imagery."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .detector import decide
from .evaluate import UNKNOWN, Annotation, Prediction, evaluate, write_predictions, write_truth
from .pilot_data import PROXY_UNKNOWN, PilotTiles, collate, ground_truth, make_pilot_split
from .prototype_gate import decide_by_prototype, fit_prototype_features, prototype_distances
from .real_model import STRIDES, RealDetector, assignments, detection_loss, predict_candidates
from .round1_data import read_class_map
from .synthetic import _unknown_proposal_recall


def _loader(args, crops, device, *, shuffle=False, distributed=False):
    dataset = PilotTiles(args.root, crops, args.crop_size)
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=args.seed) if distributed else None
    return DataLoader(dataset, batch_size=args.batch, collate_fn=collate, sampler=sampler,
                      shuffle=shuffle and sampler is None, pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(args.seed))


def _save_checkpoint(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train(model, loader, classes, epochs, lr, device, lead, save_epoch, seed, resume):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    active = torch.tensor(classes, device=device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    timings = []
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        timings = list(resume["epoch_seconds"])
    for epoch in range(len(timings), epochs):
        started = time.perf_counter()
        model.train()
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        loader.generator.manual_seed(seed + epoch)
        total = torch.zeros((), device=device)
        for images, labels, background, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                outputs, _ = model(images.to(device, non_blocking=True))
                loss = detection_loss(outputs, labels, background.to(device), active)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach()
        mean_loss = total.item() / len(loader)
        timings.append(time.perf_counter() - started)
        if lead:
            print(f"  epoch {epoch + 1}/{epochs} loss {mean_loss:.3f} in {timings[-1]:.1f}s", flush=True)
            save_epoch(optimizer, scaler, timings)
    return timings


@torch.inference_mode()
def _fit(model, loader, classes, device):
    model.eval()
    feats = defaultdict(list)
    for images, labels, _, _ in loader:
        _, maps = model(images.to(device))
        for level, (feature, stride) in enumerate(zip(maps, STRIDES, strict=True)):
            assigned = assignments(labels, feature.shape[-2:], stride, level, device)
            if assigned is None:
                continue
            indices, class_ids, _ = assigned
            embeddings = feature.permute(0, 2, 3, 1)[indices].float().cpu()
            for class_id, embedding in zip(class_ids.tolist(), embeddings, strict=True):
                feats[class_id].append(embedding)
    represented = [torch.stack(feats[class_id]) for class_id in classes if feats[class_id]]
    return fit_prototype_features(represented), [key for key in classes if not feats[key]]


def _extras(candidates, floor):
    per_tile = defaultdict(int)
    extra = []
    for row in sorted(candidates, key=lambda candidate: -candidate.objectness):
        if row.objectness < floor and per_tile[row.image_id] < 5:
            extra.append(row)
            per_tile[row.image_id] += 1
    return extra


def _extra_flags(extras, distances, floor, cutoff):
    return [Prediction(row.image_id, UNKNOWN, row.box, row.objectness * distance / (distance + 1))
            for row, distance in zip(extras, distances, strict=True)
            if row.objectness >= floor and distance > cutoff]


def _quantiles(values, probabilities):
    return np.unique(np.quantile(values, probabilities)).tolist()


def _pick(configurations, predict, truth, classes):
    def key(config):
        scores = evaluate(predict(*config), truth, classes)
        return scores["OE-Score"], scores["mAP_K"], scores["U-F1"]
    return max(configurations, key=key)


def _calibrate(candidates, prototypes, truth, classes):
    objects = [row.objectness for row in candidates]
    confidence = [row.class_probability for row in candidates]
    floor, cls_cutoff = _pick(((floor, cutoff) for floor in _quantiles(objects, (0.5, 0.75, 0.9, 0.97))
                               for cutoff in [0.0, *_quantiles(confidence, (0.25, 0.5, 0.75, 0.9))]),
                              lambda f, c: decide(candidates, f, c), truth, classes)
    normal = [row for row in candidates if row.objectness >= floor]
    distances = prototype_distances(normal, prototypes)
    distance_cutoff, = _pick(((cutoff,) for cutoff in _quantiles(distances, (0.25, 0.5, 0.7, 0.85, 0.95, 1))),
                              lambda cutoff: decide_by_prototype(normal, distances, floor, cutoff), truth, classes)
    predictions = decide_by_prototype(normal, distances, floor, distance_cutoff)
    extras = _extras(candidates, floor)
    if not extras:
        return floor, cls_cutoff, distance_cutoff, None
    distances = prototype_distances(extras, prototypes)
    extra_floor, extra_distance = _pick(
        ((minimum, cutoff) for minimum in [0.0, *_quantiles([row.objectness for row in extras], (0.5, 0.85))]
         for cutoff in {distance_cutoff, *[max(distance_cutoff, q) for q in _quantiles(distances, (0.5, 0.8, 0.95, 1))]}),
        lambda f, d: predictions + _extra_flags(extras, distances, f, d), truth, classes,
    )
    extra = _extra_flags(extras, distances, extra_floor, extra_distance)
    enabled = evaluate(predictions + extra, truth, classes)["OE-Score"] > evaluate(predictions, truth, classes)["OE-Score"]
    return floor, cls_cutoff, distance_cutoff, {"floor": extra_floor, "distance": extra_distance, "enabled": enabled}


def _evaluate_stage(args, model, device, stage, splits, classes, class_map, timings):
    parts = [splits[name] for name in ("A", "B") if stage == "B" or name == "A"]
    support, calibration, validation = ([crop for part in parts for crop in part[key]]
                                        for key in ("training", "calibration", "validation"))
    names = tuple(class_map[class_id] for class_id in classes)
    prototypes, missing = _fit(model, _loader(args, support, device), classes, device)
    tune = predict_candidates(model, _loader(args, calibration, device), classes, class_map, device, args.proposals)
    floor, cls_cutoff, distance_cutoff, extra = _calibrate(tune, prototypes, ground_truth(calibration, class_map), names)
    candidates = predict_candidates(model, _loader(args, validation, device), classes, class_map, device, args.proposals)
    truth = ground_truth(validation, class_map)
    normal = [row for row in candidates if row.objectness >= floor]
    primary = decide_by_prototype(normal, prototype_distances(normal, prototypes), floor, distance_cutoff)
    methods = {"softmax": decide(normal, floor, cls_cutoff), "prototype": primary}
    if extra is not None:
        extras = _extras(candidates, floor)
        methods["prototype_extra"] = primary + _extra_flags(extras, prototype_distances(extras, prototypes),
                                                           extra["floor"], extra["distance"])
    write_truth(args.output / f"stage_{stage}_truth.csv", truth)
    scores = {}
    for method, rows in methods.items():
        write_predictions(args.output / f"stage_{stage}_{method}.csv", rows)
        scores[method] = evaluate(rows, truth, names)
        score = scores[method]
        print(f"  {method}: OE={score['OE-Score']:.2f}, mAP={score['mAP_K']:.4f}, U-F1={score['U-F1']:.4f}", flush=True)
    all_as_unknown = [Annotation(row.image_id, UNKNOWN, row.box) for row in truth]
    raw_recall = _unknown_proposal_recall(candidates, all_as_unknown, 0.0)
    print(f"  raw box recall at IoU 0.5: {raw_recall:.1%}", flush=True)
    return {"training_seconds": sum(timings), "epoch_seconds": timings,
            "calibration_crops": len(calibration), "validation_crops": len(validation),
            "active_class_ids": list(classes), "missing_prototypes": missing, "objectness_floor": floor,
            "softmax_cutoff": cls_cutoff, "prototype_distance_cutoff": distance_cutoff, "extra": extra,
            "raw_box_recall": raw_recall,
            "raw_unknown_box_recall": _unknown_proposal_recall(candidates, truth, 0.0),
            "unknown_proposal_recall": _unknown_proposal_recall(candidates, truth, floor), "methods": scores}


def run(args, device, rank, multi):
    lead = rank == 0
    torch.manual_seed(args.seed)
    splits = make_pilot_split(args.root, args.crop_size, args.base_tiles, args.new_tiles, args.eval_tiles, args.seed)
    manifest = {split: {name: [{"tile_id": crop.tile_id, "left": crop.left, "top": crop.top} for crop in rows]
                        for name, rows in roles.items()}
                for split, roles in splits.items()}
    class_map = read_class_map(args.root)
    signature = sha256(json.dumps((args.crop_size, args.seed, manifest, class_map), sort_keys=True).encode()).hexdigest()
    if lead:
        args.output.mkdir(parents=True, exist_ok=True)
    base = tuple(class_id for class_id in range(30) if class_id not in PROXY_UNKNOWN)
    novel = tuple(range(30, 40))
    model = RealDetector().to(device)
    wrapped = model
    if multi:
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    old = splits["A"]["training"]
    replay = old[::2]
    incremental = replay + splits["B"]["training"] * 2
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    report = {"device": str(device), "processes": world_size, "seed": args.seed, "split_signature": signature,
              "crop_size": args.crop_size, "proxy_unknown_ids": PROXY_UNKNOWN,
              "excluded_set_b_tile_ids_from_set_a": True, "stages": {}}
    metrics_path = args.output / "metrics.json"
    if lead and args.stage == "B" and args.stage_a is None and metrics_path.is_file():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous.get("split_signature") == signature and "A" in previous["stages"]:
            report["stages"]["A"] = previous["stages"]["A"]
    for stage, training, classes, epochs, lr in (("A", old, base, args.epochs_a, 0.002),
                                                  ("B", incremental, base + novel, args.epochs_b, 0.001)):
        if args.stage not in ("both", stage):
            continue
        metadata = {"stage": stage, "active_class_ids": list(classes), "crop_size": args.crop_size,
                    "split_signature": signature}
        train_config = {"batch": args.batch, "processes": world_size, "device_type": device.type}
        resume = None
        if args.resume:
            resume = torch.load(args.resume, map_location=device, weights_only=True)
            if (any(resume.get(key) != value for key, value in metadata.items())
                    or resume.get("train_config") != train_config or "optimizer" not in resume
                    or len(resume["epoch_seconds"]) > epochs):
                raise ValueError("resume requires a matching stage, split, device, GPU count, batch size and epoch budget")
            model.load_state_dict(resume["model"])
        elif stage == "B":
            stage_a_path = args.stage_a or args.output / "stage_A.pt"
            checkpoint = torch.load(stage_a_path, map_location=device, weights_only=True)
            if (checkpoint.get("stage") != "A" or checkpoint.get("split_signature") != signature
                    or checkpoint.get("active_class_ids") != list(base)):
                raise ValueError("Stage B requires a Stage A checkpoint with matching pilot split and classes")
            model.load_state_dict(checkpoint["model"])

        def save_epoch(optimizer, scaler, timings):
            _save_checkpoint({**metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(), "epoch_seconds": timings, "train_config": train_config},
                             args.output / f"stage_{stage}_last.pt")

        if lead:
            (args.output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Stage {stage}: training on {len(training)} crops per epoch", flush=True)
        timings = _train(wrapped, _loader(args, training, device, shuffle=True, distributed=multi),
                         classes, epochs, lr, device, lead, save_epoch, args.seed, resume)
        if lead:
            _save_checkpoint({**metadata, "model": model.state_dict()}, args.output / f"stage_{stage}.pt")
            report["stages"][stage] = _evaluate_stage(args, model, device, stage, splits, classes, class_map, timings)
            metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if multi:
            dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("round1"))
    parser.add_argument("--output", type=Path, default=Path("runs/real_pilot"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stage", choices=("A", "B", "both"), default="both")
    parser.add_argument("--stage-a", type=Path, help="Stage A checkpoint to initialize a separate Stage B run")
    parser.add_argument("--resume", type=Path, help="Resume one stage from its stage_*_last.pt checkpoint")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--crop-size", type=int, choices=(256, 384, 512), default=256)
    parser.add_argument("--base-tiles", type=int, default=6, help="Target training crops per known base class")
    parser.add_argument("--new-tiles", type=int, default=4, help="Target training crops per new class")
    parser.add_argument("--eval-tiles", type=int, default=2, help="Crops per class for each of calibration and validation")
    parser.add_argument("--epochs-a", type=int, default=4)
    parser.add_argument("--epochs-b", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8, help="Batch size per process")
    parser.add_argument("--proposals", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.resume and (args.stage == "both" or args.stage_a):
        parser.error("--resume requires one stage and supplies its own starting weights")
    if args.stage_a and args.stage != "B":
        parser.error("--stage-a requires --stage B")
    if min(args.base_tiles, args.new_tiles, args.eval_tiles, args.epochs_a, args.epochs_b,
           args.batch, args.proposals, args.threads) < 1:
        parser.error("counts must be positive")
    use_gpu = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_gpu and not torch.cuda.is_available():
        parser.error("CUDA requested but this PyTorch installation has no available CUDA device")
    multi = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if use_gpu else torch.device("cpu")
    torch.set_num_threads(args.threads)
    if use_gpu:
        torch.cuda.set_device(device)
    if multi:
        dist.init_process_group(backend="nccl" if use_gpu else "gloo")
    try:
        run(args, device, rank, multi)
    finally:
        if multi:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
