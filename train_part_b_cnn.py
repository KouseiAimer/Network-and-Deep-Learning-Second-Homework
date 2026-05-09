import gzip
import json
from pathlib import Path
from struct import unpack

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


SEED = 309
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.08
WEIGHT_DECAY = 1e-4
CONV_CHANNELS = 16
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
HIDDEN_DIM = 64


def load_images(path):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images.astype(np.float64) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = unpack(">2I", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def initialize_he(model):
    conv_fan_in = model.conv.in_channels * model.conv.kernel_size[0] * model.conv.kernel_size[1]
    model.conv.W[:] = np.random.normal(
        0.0,
        np.sqrt(2.0 / conv_fan_in),
        size=model.conv.W.shape,
    )
    model.conv.b[:] = 0.0

    fc1_fan_in = model.fc1.W.shape[0]
    model.fc1.W[:] = np.random.normal(0.0, np.sqrt(2.0 / fc1_fan_in), size=model.fc1.W.shape)
    model.fc1.b[:] = 0.0

    fc2_fan_in = model.fc2.W.shape[0]
    model.fc2.W[:] = np.random.normal(0.0, np.sqrt(2.0 / fc2_fan_in), size=model.fc2.W.shape)
    model.fc2.b[:] = 0.0


def evaluate(model, loss_fn, images, labels, batch_size=1024):
    total_loss = 0.0
    total_correct = 0
    total = labels.shape[0]

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits = model(images[start:end])
        loss = loss_fn(logits, labels[start:end])
        preds = np.argmax(logits, axis=1)
        total_loss += loss * (end - start)
        total_correct += np.sum(preds == labels[start:end])

    return total_loss / total, total_correct / total


def plot_learning_curve(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    axes[0].plot(history["epoch"], history["valid_loss"], marker="s", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_acc"], marker="o", label="Train")
    axes[1].plot(history["epoch"], history["valid_acc"], marker="s", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CNN Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_filters(model, output_path):
    weights = model.conv.W[:, 0, :, :]
    num_filters = weights.shape[0]
    cols = 4
    rows = int(np.ceil(num_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < num_filters:
            ax.imshow(weights[i], cmap="gray")
            ax.set_title(f"Filter {i}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    np.random.seed(SEED)

    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent
    output_dir = project_root / "PartB"
    weights_dir = project_root / "model_weights_for_upload"
    output_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    data_dir = code_dir / "dataset" / "MNIST"
    train_images = load_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    split_path = project_root / "PartA" / "parta_split_indices.npy"
    if split_path.exists():
        indices = np.load(split_path)
    else:
        indices = np.random.permutation(train_images.shape[0])
        np.save(output_dir / "partb_split_indices.npy", indices)

    train_images = train_images[indices]
    train_labels = train_labels[indices]
    valid_images = train_images[:10000]
    valid_labels = train_labels[:10000]
    train_images = train_images[10000:]
    train_labels = train_labels[10000:]

    model = nn.models.Model_CNN(
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        hidden_dim=HIDDEN_DIM,
        weight_decay=True,
        weight_decay_lambda=WEIGHT_DECAY,
    )
    initialize_he(model)

    optimizer = nn.optimizer.SGD(init_lr=LEARNING_RATE, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    best_valid_acc = -1.0
    best_model_path = weights_dir / "partb_cnn_best_model.pickle"
    num_train = train_labels.shape[0]

    for epoch in range(1, EPOCHS + 1):
        order = np.random.permutation(num_train)
        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, num_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_train)
            batch_idx = order[start:end]
            batch_x = train_images[batch_idx]
            batch_y = train_labels[batch_idx]

            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            batch_acc = nn.metric.accuracy(logits, batch_y)

            loss_fn.backward()
            optimizer.step()

            epoch_loss += loss * (end - start)
            epoch_correct += batch_acc * (end - start)

        train_loss = epoch_loss / num_train
        train_acc = epoch_correct / num_train
        valid_loss, valid_acc = evaluate(model, loss_fn, valid_images, valid_labels)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["valid_loss"].append(float(valid_loss))
        history["valid_acc"].append(float(valid_acc))

        print(
            f"epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            model.save_model(best_model_path)

    best_model = nn.models.Model_CNN()
    best_model.load_model(best_model_path)
    best_loss_fn = nn.op.MultiCrossEntropyLoss(model=best_model, max_classes=10)
    test_loss, test_acc = evaluate(best_model, best_loss_fn, test_images, test_labels)

    plot_learning_curve(history, output_dir / "cnn_learning_curve.png")
    plot_filters(best_model, output_dir / "cnn_filters.png")

    summary = {
        "seed": SEED,
        "model": f"Conv2D(1->{CONV_CHANNELS}, k={KERNEL_SIZE}, stride={STRIDE}, padding={PADDING}) + ReLU + Linear({HIDDEN_DIM}) + ReLU + Linear",
        "input_shape": "[batch, 1, 28, 28]",
        "conv_output_shape": list(best_model.conv_output_shape),
        "flatten_dim": best_model.flatten_dim,
        "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "SGD",
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "None",
        "train_size": int(train_labels.shape[0]),
        "valid_size": int(valid_labels.shape[0]),
        "test_size": int(test_labels.shape[0]),
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_valid_loss": history["valid_loss"][-1],
        "final_valid_acc": history["valid_acc"][-1],
        "best_valid_acc": float(best_valid_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "best_model_path": str(best_model_path.relative_to(project_root)),
        "mlp_part_a_valid_acc": 0.9291,
        "mlp_part_a_test_acc": 0.9318,
    }

    with open(output_dir / "partb_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "history": history}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
