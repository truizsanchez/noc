import random

import pytest

from noc.ch10_nn.perceptron import EvolvedPerceptrons, Perceptron, TrainingSet, accuracy, line


def test_feedforward_is_the_sign_of_the_weighted_sum() -> None:
    p = Perceptron([1.0, -2.0, 0.5])
    assert p.feedforward([3, 1, 1]) == 1  # 3 - 2 + 0.5
    assert p.feedforward([0, 1, 1]) == -1


def test_training_moves_weights_by_error_times_input() -> None:
    p = Perceptron([0.0, 0.0, -1.0], learning_rate=0.1)  # guesses -1 for this point
    p.train([2.0, 3.0, 1.0], 1)  # error = 2
    assert p.weights == pytest.approx([0.4, 0.6, -0.8])


def test_boundary_line() -> None:
    p = Perceptron([0.5, -1.0, 1.0])  # 0.5x - y + 1 = 0  ->  y = 0.5x + 1
    assert p.boundary(4) == pytest.approx(line(4))


@pytest.mark.parametrize(("normalize", "rate"), [(False, 0.0001), (True, 0.01)])
def test_perceptron_learns_the_line(normalize: bool, rate: float) -> None:
    rng = random.Random(1)
    raw = TrainingSet.random(2000, 640, 240, rng)
    answers = [raw.answer(point, line) for point in raw.points]
    data = raw.normalized(640, 240) if normalize else raw
    p = Perceptron.random(3, rate, rng)
    for _ in range(10):
        for point, answer in zip(data.points, answers, strict=True):
            p.train(point, answer)
    assert accuracy(p, data, answers) > 0.97


def test_normalized_points_fit_in_the_unit_square() -> None:
    data = TrainingSet.random(100, 640, 240, random.Random(2)).normalized(640, 240)
    assert all(-1 <= x <= 1 and -1 <= y <= 1 and b == 1 for x, y, b in data.points)


def test_genetic_algorithm_finds_good_weights() -> None:
    rng = random.Random(3)
    raw = TrainingSet.random(500, 640, 240, rng)
    answers = [raw.answer(point, line) for point in raw.points]
    evolution = EvolvedPerceptrons(raw.normalized(640, 240), answers, rng=rng)
    first = accuracy(evolution.best, evolution.data, answers)
    for _ in range(40):
        evolution.evolve()
    assert accuracy(evolution.best, evolution.data, answers) >= max(first, 0.9)
