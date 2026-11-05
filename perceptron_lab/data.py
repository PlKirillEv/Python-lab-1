"""Данные: синтетические точки и локальная копия iris."""

import csv
import random
from pathlib import Path


def make_data(n_samples: int, seed: int = 42) -> tuple[list[list[float]], list[int]]:
    """Генерирует два линейно разделимых класса вокруг центров (-2,-2) и (2,2)."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(n_samples):
        label = i % 2
        center = 2.0 if label == 1 else -2.0
        X.append([rng.gauss(center, 0.8), rng.gauss(center, 0.8)])
        y.append(label)
    return X, y


def load_iris() -> tuple[list[list[float]], list[int]]:
    """Читает iris (setosa и versicolor, признаки - длина и ширина лепестка)."""
    path = Path(__file__).parent / "data" / "iris_2class.csv"
    X: list[list[float]] = []
    y: list[int] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row["petal_length"]), float(row["petal_width"])])
            y.append(int(row["label"]))
    return X, y
