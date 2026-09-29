# Chapter 10 — Neural Networks

Run: `uv run python -m noc.ch10_nn` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Example 10.1 | a perceptron learning which side of a line points are on; its own line in red (Exercise 10.1), `N` normalizes the inputs (Exercise 10.3) |
| 2 | Exercise 10.2 | the same perceptron with weights found by a genetic algorithm |
| 3 | Putting the "Network" in Neural Network | a two-layer network learning XOR, with its output over the whole input square |
| 4 | Example 10.2 | the gesture classifier: training with a loss graph, then classifying mouse strokes; hold `U`/`D`/`L`/`R` while dragging to add examples (Exercise 10.4) |

## Book sections → code

| Book section | Code |
|---|---|
| The Perceptron; The Perceptron Code | `ch10_nn.perceptron.Perceptron` (`feedforward`, `train`), `TrainingSet`, `line` |
| Exercises 10.1-10.3 | `Perceptron.boundary`, `EvolvedPerceptrons` (chapter 9's GA), `TrainingSet.normalized` |
| Putting the "Network" in Neural Network (XOR, backpropagation) | `noc.common.neural.NeuralNetwork` |
| Machine Learning with ml5.js: network design, training, evaluation | `NeuralNetwork.train_step`, `loss`; `Normalizer` (`normalizeData()`) |
| Building a Gesture Classifier; deployment | `Classifier` (`train_epoch`, `classify`); `NeuralNetwork.to_json`/`from_json` (Exercise 10.5's `save()`/`load()`) |

## Design decisions and deviations

- **numpy instead of ml5.js.** The book deliberately leaves backpropagation to ml5.js (and
  TensorFlow.js) and points to the Coding Train's "Toy Neural Network" for the details. Here
  the network is written out in about 150 lines of numpy: layers of `inputs @ weights + bias`,
  sigmoid or ReLU hidden layers, a softmax, sigmoid or linear output, and gradient descent.
  The tests compare every layer's backpropagated gradient with a numerical one.
- **One gradient for three losses.** Softmax with cross-entropy, sigmoid with binary
  cross-entropy and a linear output with squared error all have the gradient
  `prediction - target` at the output layer, so `train_step` handles classification, yes/no
  outputs and regression with the same code.
- **`Classifier` mirrors `ml5.neuralNetwork({task: "classification"})`**: it takes labels,
  normalizes inputs to [0, 1] from the training data like `normalizeData()`, uses a hidden
  layer of 16 neurons like ml5's default, and `classify()` returns `(label, confidence)` pairs
  sorted by confidence. Training runs a couple of epochs per frame, and the sketch draws the
  loss curve the way ml5's debug Visor does.
- **Normalization and learning rate (Exercise 10.3).** The book's perceptron learns from raw
  pixel coordinates with a learning rate of 0.0001; normalized to [-1, 1], the inputs are about
  300 times smaller and the same accuracy comes with a rate of 0.01.
- **Drawing 2,000 training points** per frame uses `Canvas.discs`, which draws same-size
  outlined circles as tinted sprites (35 → 4 ms per frame).
- **The XOR sketch** is not a numbered example: it makes the book's argument for hidden layers
  visible (a perceptron can't separate XOR; two layers can).
- Not ported: Exercises 10.6 and 10.7 (other ml5.js models: Handpose, image classification).
  Exercise 10.4 adds examples live instead of saving a JSON file.

## Tests

`tests/common/test_neural.py` checks the backpropagated gradients against numerical ones for
all three output types, XOR learning, copying, mutation, crossover, JSON round trips, the
normalizer and the gesture classifier. `tests/ch10_nn` checks the perceptron's update rule
and boundary, that it learns the line with raw and normalized inputs, and that the GA finds
good weights.
