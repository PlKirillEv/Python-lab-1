"""Графики: граница решения и ошибки по эпохам."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perceptron_lab.model import Perceptron


def plot_boundary(
    model: Perceptron, X: list[list[float]], y: list[int], path: str, title: str
) -> None:
    """Рисует точки двух классов и линию w1*x1 + w2*x2 + bias = 0."""
    for cls, color in ((0, "tab:blue"), (1, "tab:red")):
        xs = [p[0] for p, t in zip(X, y) if t == cls]
        ys = [p[1] for p, t in zip(X, y) if t == cls]
        plt.scatter(xs, ys, c=color, label=f"класс {cls}")

    w1, w2 = model.weights
    x_min = min(p[0] for p in X)
    x_max = max(p[0] for p in X)
    if w2 != 0:
        line_y = [-(w1 * x1 + model.bias) / w2 for x1 in (x_min, x_max)]
        plt.plot([x_min, x_max], line_y, "k-", label="граница решения")
    y_min = min(p[1] for p in X)
    y_max = max(p[1] for p in X)
    plt.ylim(y_min - 0.5, y_max + 0.5)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_errors(model: Perceptron, path: str) -> None:
    """Рисует, как уменьшается число ошибок от эпохи к эпохе."""
    epochs = list(range(1, len(model.errors_per_epoch) + 1))
    plt.plot(epochs, model.errors_per_epoch, marker="o")
    plt.xlabel("Эпоха")
    plt.ylabel("Число ошибок")
    plt.title("Ошибки по эпохам")
    plt.savefig(path)
    plt.close()
