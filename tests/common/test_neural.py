import numpy as np
import pytest

from noc.common.neural import Classifier, NeuralNetwork, Normalizer, sigmoid, softmax


def numeric_gradient(net: NeuralNetwork, x: np.ndarray, y: np.ndarray, layer: int) -> np.ndarray:
    grad = np.zeros_like(net.weights[layer])
    eps = 1e-6
    for idx in np.ndindex(grad.shape):
        a, b = net.copy(), net.copy()
        a.weights[layer][idx] += eps
        b.weights[layer][idx] -= eps
        grad[idx] = (a.loss(x, y) - b.loss(x, y)) / (2 * eps)
    return grad


@pytest.mark.parametrize(
    ("output", "hidden"), [("softmax", "sigmoid"), ("sigmoid", "relu"), ("linear", "sigmoid")]
)
def test_backpropagation_matches_numeric_gradients(output: str, hidden: str) -> None:
    rng = np.random.default_rng(0)
    net = NeuralNetwork.create((3, 5, 4, 2), rng, output=output, hidden=hidden)  # type: ignore[arg-type]
    x = rng.random((6, 3))
    y = np.eye(2)[rng.integers(0, 2, 6)] if output != "linear" else rng.random((6, 2))
    stepped = net.copy()
    lr = 1e-6
    stepped.train_step(x, y, lr)
    for layer in range(3):
        analytic = (net.weights[layer] - stepped.weights[layer]) / lr
        assert analytic == pytest.approx(numeric_gradient(net, x, y, layer), abs=1e-6)


def test_activations() -> None:
    assert sigmoid(np.array([0.0]))[0] == 0.5
    probs = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert probs.sum() == pytest.approx(1)
    assert np.argmax(probs) == 2


def test_learns_xor() -> None:
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    net = NeuralNetwork.create((2, 4, 1), np.random.default_rng(1), output="sigmoid")
    for _ in range(5000):
        net.train_step(x, y, 2.0)
    assert np.round(net.predict(x)).tolist() == y.tolist()


def test_copy_is_independent_and_mutation_changes_some_weights() -> None:
    rng = np.random.default_rng(2)
    net = NeuralNetwork.create((4, 8, 2), rng)
    twin = net.copy()
    twin.mutate(0.1, rng)
    changed = sum((a != b).sum() for a, b in zip(net.weights, twin.weights, strict=True))
    total = sum(w.size for w in net.weights)
    assert 0 < changed < total * 0.3


def test_crossover_takes_each_weight_from_a_parent() -> None:
    rng = np.random.default_rng(3)
    a = NeuralNetwork.create((3, 4, 2), rng)
    b = NeuralNetwork.create((3, 4, 2), rng)
    child = a.crossover(b, rng)
    for wc, wa, wb in zip(child.weights, a.weights, b.weights, strict=True):
        assert ((wc == wa) | (wc == wb)).all()
        assert (wc == wa).any()
        assert (wc == wb).any()


def test_json_round_trip() -> None:
    net = NeuralNetwork.create((2, 3, 2), np.random.default_rng(4), output="sigmoid")
    loaded = NeuralNetwork.from_json(net.to_json())
    x = np.array([0.3, -0.2])
    assert loaded.predict(x) == pytest.approx(net.predict(x))
    assert loaded.output == "sigmoid"


def test_normalizer() -> None:
    norm = Normalizer.fit(np.array([[0.0, 10.0], [4.0, 10.0]]))
    assert norm([2.0, 10.0]).tolist() == [0.5, 0.0]


def test_gesture_classifier() -> None:
    data = [
        (0.99, 0.02, "right"),
        (0.76, -0.1, "right"),
        (-1.0, 0.12, "left"),
        (-0.9, -0.1, "left"),
        (0.02, 0.98, "down"),
        (-0.2, 0.75, "down"),
        (0.01, -0.9, "up"),
        (-0.1, -0.8, "up"),
    ]
    inputs = [(x, y) for x, y, _ in data]
    labels = [label for *_, label in data]
    classifier = Classifier.create(inputs, labels, rng=np.random.default_rng(5))
    losses = [classifier.train_epoch(inputs, labels, 1.0) for _ in range(200)]
    assert losses[-1] < losses[0] / 5
    results = classifier.classify((0.0, -1.0))
    assert results[0][0] == "up"
    assert sum(conf for _, conf in results) == pytest.approx(1)
    assert classifier.classify((1.0, 0.1))[0][0] == "right"
