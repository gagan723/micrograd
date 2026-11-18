"""Train a linear model to learn the function y = 2x + 1.

Run this file from the project directory:

    python train_regression.py
"""

import random

from nn import MLP


TRAINING_INPUTS = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
TRAINING_TARGETS = [-3.0, -1.0, 1.0, 3.0, 5.0]


def train(epochs=100, learning_rate=0.05):
    """Create a linear model, train it, and return the trained model."""
    random.seed(42)
    model = MLP(1, [1])

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

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | loss: {loss.data:.6f}")

    return model


def show_predictions(model):
    """Print the model's prediction alongside each expected value."""
    print("\nLinear regression predictions:")
    for features, target in zip(TRAINING_INPUTS, TRAINING_TARGETS):
        prediction = model(features).data
        print(
            f"x={features[0]: 4.1f} prediction={prediction: 7.4f} "
            f"target={target: 4.1f}"
        )

    weight, bias = model.parameters()
    print(f"\nLearned function: y = {weight.data:.4f}x + {bias.data:.4f}")


if __name__ == "__main__":
    trained_model = train()
    show_predictions(trained_model)
