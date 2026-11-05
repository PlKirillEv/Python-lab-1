from perceptron_lab.data import load_iris, make_data
from perceptron_lab.model import Perceptron


def test_activate_threshold() -> None:
    p = Perceptron(2)
    assert p.activate(0) == 1
    assert p.activate(-1) == 0
    assert p.activate(3.5) == 1


def test_weights_length() -> None:
    assert len(Perceptron(5).weights) == 5


def test_predict_with_manual_weights() -> None:
    p = Perceptron(2)
    p.weights = [1.0, 1.0]
    p.bias = -1.0
    assert p.predict([2.0, 2.0]) == 1
    assert p.predict([-2.0, -2.0]) == 0


def test_train_separable_data() -> None:
    X, y = make_data(100)
    p = Perceptron(2, learning_rate=0.1, epochs=50)
    p.train(X, y)
    assert p.score(X, y) > 0.95


def test_train_iris() -> None:
    X, y = load_iris()
    p = Perceptron(2, learning_rate=0.1, epochs=100)
    p.train(X, y)
    assert p.score(X, y) == 1.0


def test_errors_recorded() -> None:
    X, y = make_data(50)
    p = Perceptron(2, epochs=10)
    p.train(X, y)
    assert 1 <= len(p.errors_per_epoch) <= 10
