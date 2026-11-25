"""Deterministic synthetic scenes with sparse objects and background clutter."""

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .evaluate import UNKNOWN, Annotation, Box

IMAGE_SIZE = 64
CLASSES = ("circle", "square", "triangle")
BASE_CLASSES = CLASSES[:2]
CALIBRATION_SHAPES = ("pentagon", "plus")


@dataclass(frozen=True)
class Scene:
    image_id: str
    pixels: np.ndarray
    objects: tuple[tuple[str, Box], ...]


def _shape(draw, name: str, x: int, y: int, size: int, color: int) -> None:
    right, bottom = x + size - 1, y + size - 1
    mid_x, mid_y = x + size // 2, y + size // 2
    if name == "circle":
        draw.ellipse((x, y, right, bottom), fill=color)
    elif name == "square":
        draw.rectangle((x, y, right, bottom), fill=color)
    elif name == "triangle":
        draw.polygon(((mid_x, y), (right, bottom), (x, bottom)), fill=color)
    elif name == "diamond":
        draw.polygon(((mid_x, y), (right, mid_y), (mid_x, bottom), (x, mid_y)), fill=color)
    elif name == "pentagon":
        delta = size // 4
        draw.polygon(((mid_x, y), (right, mid_y), (right - delta, bottom),
                      (x + delta, bottom), (x, mid_y)), fill=color)
    elif name == "plus":
        delta = size // 3
        draw.rectangle((x + delta, y, right - delta, bottom), fill=color)
        draw.rectangle((x, y + delta, right, bottom - delta), fill=color)
    else:
        raise ValueError(f"unsupported shape: {name}")


def _place_shape(rng, draw, name, objects, image_id) -> Box:
    for _ in range(500):
        size = rng.randint(10, 19)
        x, y = (rng.randrange(1, IMAGE_SIZE - size) for _ in range(2))
        box = (x, y, x + size, y + size)
        if all(box[2] + 2 <= left or box[0] >= right + 2 or
               box[3] + 2 <= top or box[1] >= bottom + 2
               for _, (left, top, right, bottom) in objects):
            _shape(draw, name, x, y, size, rng.randint(140, 250))
            return box
    raise RuntimeError(f"could not fit {name} in {image_id}")


def _make_scene(rng, noise, image_id: str, names: tuple[str, ...]) -> Scene:
    background = noise.normal(rng.randint(19, 55), 12, (IMAGE_SIZE, IMAGE_SIZE)).clip(0, 255)
    image = Image.fromarray(background.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(2, 9)):
        x, y = rng.randrange(IMAGE_SIZE), rng.randrange(IMAGE_SIZE)
        draw.line((x, y, x + rng.randint(1, 5), y + rng.randint(-3, 3)), fill=rng.randint(65, 120))

    objects = []
    for name in names:
        objects.append((name, _place_shape(rng, draw, name, objects, image_id)))
    return Scene(image_id, np.array(image), tuple(objects))


def make_split(seed: int, split: str, count: int) -> list[Scene]:
    rng = random.Random(f"{seed}:{split}")
    noise = np.random.default_rng(rng.getrandbits(64))
    scenes = []
    for index in range(count):
        if split == "few_shot":
            names = ("triangle",)
        else:
            names = tuple(rng.choices(BASE_CLASSES, k=rng.choices((0, 1, 2), (1, 3, 3))[0]))
            if split == "calibration" and index % 2 == 0:
                names += (rng.choice(CALIBRATION_SHAPES),)
            elif split == "test" and index % 3 != 0:
                names += (("triangle", "diamond")[index % 2],)
        scenes.append(_make_scene(rng, noise, f"{split}_{index:05d}", names))
    return scenes


def annotations(scenes: list[Scene], classes: tuple[str, ...]) -> list[Annotation]:
    return [Annotation(scene.image_id, name if name in classes else UNKNOWN, box)
            for scene in scenes for name, box in scene.objects]


def add_unlabelled_diamonds(scenes: list[Scene], seed: int, count: int) -> list[Scene]:
    """Add one non-overlapping diamond to each of a fixed number of training tiles."""
    rng = random.Random(f"{seed}:unlabelled_diamonds")
    selected = set(rng.sample(range(len(scenes)), count))
    modified = list(scenes)
    for index in sorted(selected):
        scene = scenes[index]
        image = Image.fromarray(scene.pixels.copy())
        _place_shape(rng, ImageDraw.Draw(image), "diamond", scene.objects, scene.image_id)
        modified[index] = Scene(scene.image_id, np.array(image), scene.objects)
    return modified
