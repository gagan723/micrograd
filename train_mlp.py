"""Build and train a small multilayer perceptron with micrograd.

Run this file from the project directory:

    python train_mlp.py
"""

import random

from nn import MLP


# Each row has three input features. The target is either -1 or 1.
TRAINING_INPUTS = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
TRAINING_TARGETS = [1.0, -1.0, -1.0, 1.0]


def train(epochs=200, learning_rate=0.01):
    """Create an MLP, train it, and return the trained model."""
    random.seed(42)
    model = MLP(3, [4, 4, 1])

    print(model)
    print(f"Trainable parameters: {len(model.parameters())}\n")

    for epoch in range(epochs):
        # Forward pass: get predictions and calculate mean squared error.
        predictions = [model(features) for features in TRAINING_INPUTS]
        loss = sum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, TRAINING_TARGETS)
        ) / len(TRAINING_INPUTS)

        # Backward pass: calculate every parameter's gradient.
        model.zero_grad()
        loss.backward()

        # Gradient-descent update.
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | loss: {loss.data:.6f}")

    return model


def show_predictions(model):
    """Print the trained model's output for each training example."""
    print("\nFinal predictions:")
    for features, target in zip(TRAINING_INPUTS, TRAINING_TARGETS):
        prediction = model(features).data
        predicted_class = 1 if prediction >= 0 else -1
        print(
            f"input={str(features):20s} "
            f"prediction={prediction: .4f} "
            f"class={predicted_class:2d} target={int(target):2d}"
        )


if __name__ == "__main__":
    trained_model = train()
    show_predictions(trained_model)
