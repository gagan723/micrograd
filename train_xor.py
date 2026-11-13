"""Train a multilayer perceptron to learn the XOR function.

Run this file from the project directory:

    python train_xor.py
"""

import random

from nn import MLP


TRAINING_INPUTS = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
TRAINING_TARGETS = [-1.0, 1.0, 1.0, -1.0]


def train(epochs=500, learning_rate=0.05):
    """Create an MLP, train it on XOR, and return the trained model."""
    random.seed(42)
    model = MLP(2, [4, 4, 1])

    print(model)
    print(f"Trainable parameters: {len(model.parameters())}\n")

    for epoch in range(epochs):
        predictions = [model(features) for features in TRAINING_INPUTS]
        loss = sum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, TRAINING_TARGETS)
        ) / len(TRAINING_INPUTS)

        model.zero_grad()
        loss.backward()

        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | loss: {loss.data:.6f}")

    return model


def show_predictions(model):
    """Print the model's learned output for each XOR input."""
    print("\nXOR predictions:")
    for features, target in zip(TRAINING_INPUTS, TRAINING_TARGETS):
        prediction = model(features).data
        predicted_class = 1 if prediction >= 0 else -1
        print(
            f"input={features} prediction={prediction: .4f} "
            f"class={predicted_class:2d} target={int(target):2d}"
        )


if __name__ == "__main__":
    trained_model = train()
    show_predictions(trained_model)
