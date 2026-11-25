"""Compact multiscale detector for the real-image Vesion pilot."""

import torch
from torch import nn
from torch.nn import functional as F

from .detector import Candidate

STRIDES = (4, 8, 16)
CLASS_SLOTS = 50


def _conv(inputs, outputs, stride=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
                         nn.GroupNorm(4, outputs), nn.SiLU())


class RealDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(_conv(3, 16, 2), _conv(16, 24, 2), _conv(24, 24))
        self.middle = nn.Sequential(_conv(24, 32, 2), _conv(32, 32))
        self.deep = nn.Sequential(_conv(32, 48, 2), _conv(48, 48))
        self.lateral = nn.ModuleList(nn.Conv2d(channels, 24, 1) for channels in (24, 32, 48))
        self.shared = _conv(24, 24)
        self.head = nn.Conv2d(24, 5 + CLASS_SLOTS, 1)
        with torch.no_grad():
            self.head.bias[0] = -3.0

    def forward(self, pixels):
        high = self.stem(pixels)
        mid = self.middle(high)
        low = self.deep(mid)
        maps = [layer(feature) for layer, feature in zip(self.lateral, (high, mid, low), strict=True)]
        for level in (1, 0):
            maps[level] = maps[level] + F.interpolate(maps[level+1], size=maps[level].shape[-2:], mode="nearest")
        maps = tuple(self.shared(feature) for feature in maps)
        return tuple(self.head(feature) for feature in maps), maps


def assignments(samples, shape, stride, level, device):
    targets = {}
    for index, sample in enumerate(samples):
        for label, (x1, y1, x2, y2) in sample:
            width, height = x2 - x1, y2 - y1
            scale = max(width, height)
            if (0 if scale < 56 else 1 if scale < 128 else 2) != level:
                continue
            cx, cy = (x1 + x2) / (2 * stride), (y1 + y2) / (2 * stride)
            x, y = int(cx), int(cy)
            if not (0 <= x < shape[1] and 0 <= y < shape[0]):
                continue
            key = (index, y, x)
            if key not in targets or width * height < targets[key][1][2] * targets[key][1][3]:
                targets[key] = (label, (cx - x, cy - y, width, height))
    if not targets:
        return None
    indices = torch.tensor(list(targets), device=device).T
    labels, coordinates = zip(*targets.values())
    boxes = torch.tensor(coordinates, device=device, dtype=torch.float32)
    boxes[:, 2:] = (boxes[:, 2:] / (stride * 4)).log()
    return tuple(indices), torch.tensor(labels, device=device), boxes


def detection_loss(outputs, samples, background, class_ids):
    loss = outputs[0].new_zeros((), dtype=torch.float32)
    for level, (output, stride) in enumerate(zip(outputs, STRIDES, strict=True)):
        scores = output[:, 0].float()
        obj = torch.zeros_like(scores)
        live = background[:, None, None].expand_as(obj)
        assigned = assignments(samples, scores.shape[-2:], stride, level, output.device)
        if assigned is not None:
            indices, labels, boxes = assigned
            obj[indices] = 1
            positive = output.permute(0, 2, 3, 1)[indices].float()
            regression = torch.cat((positive[:, 1:3].sigmoid(), positive[:, 3:5]), dim=1)
            loss = loss + 2 * F.smooth_l1_loss(regression, boxes, beta=0.2)
            cls_scores = positive[:, 5 + class_ids]
            cls_targets = (labels[:, None] == class_ids).long().argmax(dim=1)
            loss = loss + F.cross_entropy(cls_scores, cls_targets)
        focal = (scores.sigmoid() - obj).square()
        supervise = live | (obj == 1)
        loss = loss + (F.binary_cross_entropy_with_logits(scores, obj, reduction="none") * focal
                       * supervise).sum() / obj.sum().clamp_min(1)
    return loss


@torch.inference_mode()
def predict_candidates(model, loader, class_ids, class_map, device, limit=200):
    from torchvision.ops import nms

    model.eval()
    active = torch.tensor(class_ids, device=device)
    predictions = []
    for images, _, _, records in loader:
        outputs, features = model(images.to(device, non_blocking=True))
        for index, record in enumerate(records):
            boxes, objects, labels, confidence, embeddings = [], [], [], [], []
            for output, feature, stride in zip(outputs, features, STRIDES, strict=True):
                data = output[index].float().permute(1, 2, 0)
                width = data.shape[1]
                scores = data[..., 0].flatten().sigmoid()
                best, positions = scores.topk(min(limit, scores.numel()), sorted=True)
                y, x = positions // width, positions % width
                values = data[y, x]
                centers = (torch.stack((x, y), dim=1) + values[:, 1:3].sigmoid()) * stride
                half = values[:, 3:5].clamp(-3, 5).exp() * (2 * stride)
                box = torch.cat((centers - half, centers + half), dim=1).clamp(0, images.shape[-1])
                probability, labels_for_level = values[:, 5 + active].softmax(dim=1).max(dim=1)
                boxes.append(box)
                objects.append(best)
                labels.append(active[labels_for_level])
                confidence.append(probability)
                embeddings.append(feature[index, :, y, x].T.float())
            boxes, objects, labels, confidence, embeddings = map(torch.cat, (boxes, objects, labels, confidence, embeddings))
            keep = nms(boxes, objects, 0.5)[:limit]
            for box, obj, label, cls, embedding in zip(boxes[keep].cpu().tolist(), objects[keep].cpu().tolist(),
                                                       labels[keep].cpu().tolist(), confidence[keep].cpu().tolist(),
                                                       embeddings[keep].cpu(), strict=True):
                predictions.append(Candidate(record.image_id, tuple(box), obj, class_map[label], cls, embedding))
    return predictions
