"""Класс Perceptron: простая модель на обычных списках и циклах for."""


class Perceptron:
    """Однослойный перцептрон для бинарной классификации (классы 0 и 1)."""

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        """Создаёт перцептрон с нулевыми весами и нулевым порогом."""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = [0.0] * n_features  # веса w
        self.bias: float = 0.0
        self.errors_per_epoch: list[int] = []

    def activate(self, z: float) -> int:
        """Пороговая функция: 1, если z >= 0, иначе 0."""
        if z >= 0:
            return 1
        return 0

    def predict(self, x: list[float]) -> int:
        """Предсказывает класс: z = w1*x1 + w2*x2 + ... + bias, затем порог."""
        z = self.bias
        for i in range(len(self.weights)):
            z += self.weights[i] * x[i]
        return self.activate(z)

    def train(self, X: list[list[float]], y: list[int]) -> None:
        """Обучает перцептрон: если ошибся, подправляем веса."""
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            errors = 0
            for idx in range(len(X)):
                x = X[idx]
                target = y[idx]
                pred = self.predict(x)
                error = target - pred
                if error != 0:
                    errors += 1
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * x[i]
                self.bias += self.learning_rate * error
            self.errors_per_epoch.append(errors)
            if errors == 0:
                break

    def score(self, X: list[list[float]], y: list[int]) -> float:
        """Возвращает долю правильных ответов (accuracy) от 0 до 1."""
        correct = 0
        for x, target in zip(X, y):
            if self.predict(x) == target:
                correct += 1
        return correct / len(y)
