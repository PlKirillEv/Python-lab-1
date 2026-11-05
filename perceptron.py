"""CLI для обучения однослойного перцептрона."""

import argparse
from pathlib import Path

from perceptron_lab.data import load_iris, make_data
from perceptron_lab.model import Perceptron
from perceptron_lab.plots import plot_boundary, plot_errors


def positive_int(value: str) -> int:
    """Проверяет, что целое число положительное."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"{value} должно быть > 0")
    return n


def positive_float(value: str) -> float:
    """Проверяет, что вещественное число положительное."""
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"{value} должно быть > 0")
    return x


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Однослойный перцептрон")
    parser.add_argument("--samples", type=positive_int, default=200, help="число точек")
    parser.add_argument(
        "--lr", type=positive_float, default=0.1, help="скорость обучения"
    )
    parser.add_argument("--epochs", type=positive_int, default=50, help="число эпох")
    parser.add_argument("--seed", type=int, default=42, help="seed генератора")
    parser.add_argument("--iris", action="store_true", help="данные iris")
    return parser.parse_args()


def run(
    X: list[list[float]], y: list[int], args: argparse.Namespace, name: str
) -> None:
    """Обучает модель и сохраняет графики."""
    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    model.train(X, y)
    print(f"[{name}] точность: {model.score(X, y):.3f}")
    print(f"[{name}] веса: {model.weights}, порог: {model.bias:.3f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    plot_boundary(
        model,
        X,
        y,
        str(results_dir / f"{name}_boundary.png"),
        f"Граница решения ({name})",
    )
    plot_errors(model, str(results_dir / f"{name}_errors.png"))


def main() -> None:
    """Точка входа: проверяет аргументы и запускает обучение."""
    args = parse_args()
    if args.iris:
        X, y = load_iris()
        run(X, y, args, "iris")
    else:
        X, y = make_data(args.samples, args.seed)
        run(X, y, args, "synthetic")


if __name__ == "__main__":
    main()
