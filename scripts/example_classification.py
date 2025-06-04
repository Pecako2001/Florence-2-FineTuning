"""Example script showing classification accuracy calculation."""
from florence.metrics import classification_accuracy


if __name__ == "__main__":
    # Example predictions and labels
    predictions = [0, 1, 1, 0]
    labels = [0, 1, 0, 0]
    acc = classification_accuracy(predictions, labels)
    print(f"Accuracy: {acc:.2f}")
