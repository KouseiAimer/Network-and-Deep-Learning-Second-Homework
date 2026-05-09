import copy
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
CONV_CHANNELS = 16
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
HIDDEN_DIM = 64

PART_B_REFERENCE = {
    "name": "cnn_sgd_baseline_reference",
    "valid_acc": 0.9585,
    "test_acc": 0.9605,
    "description": "Part B CNN baseline, SGD lr=0.08, L2=1e-4",
}


EXPERIMENTS = [
    {
        "name": "cnn_momentum",
        "category": "Optimization",
        "optimizer": "momentum",
        "learning_rate": 0.02,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "scheduler": None,
        "description": "Momentum optimizer with mu=0.9",
    },
    {
        "name": "cnn_multistep_lr",
        "category": "Optimization",
        "optimizer": "sgd",
        "learning_rate": 0.10,
        "momentum": None,
        "weight_decay": 1e-4,
        "scheduler": {"type": "multistep", "milestone_epochs": [2, 4], "gamma": 0.5},
        "description": "SGD with MultiStepLR decay after epochs 2 and 4",
    },
    {
        "name": "cnn_no_l2",
        "category": "Regularization",
        "optimizer": "sgd",
        "learning_rate": 0.08,
        "momentum": None,
        "weight_decay": 0.0,
        "scheduler": None,
        "description": "Disable L2 weight decay",
    },
    {
        "name": "cnn_strong_l2",
        "category": "Regularization",
        "optimizer": "sgd",
        "learning_rate": 0.08,
        "momentum": None,
        "weight_decay": 1e-3,
        "scheduler": None,
        "description": "Stronger L2 weight decay",
    },
]


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
    model.conv.W[:] = np.random.normal(0.0, np.sqrt(2.0 / conv_fan_in), size=model.conv.W.shape)
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


def build_scheduler(config, optimizer, steps_per_epoch):
    scheduler_config = config.get("scheduler")
    if scheduler_config is None:
        return None
    if scheduler_config["type"] == "multistep":
        milestones = [epoch * steps_per_epoch for epoch in scheduler_config["milestone_epochs"]]
        return nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=scheduler_config["gamma"],
        )
    raise ValueError(f"Unknown scheduler: {scheduler_config}")


def train_one(config, datasets, output_dir, weights_dir):
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = datasets
    weight_decay = config["weight_decay"]

    model = nn.models.Model_CNN(
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        hidden_dim=HIDDEN_DIM,
        weight_decay=weight_decay > 0,
        weight_decay_lambda=max(weight_decay, 1e-12),
    )
    initialize_he(model)

    if config["optimizer"] == "sgd":
        optimizer = nn.optimizer.SGD(init_lr=config["learning_rate"], model=model)
    elif config["optimizer"] == "momentum":
        optimizer = nn.optimizer.MomentGD(
            init_lr=config["learning_rate"],
            model=model,
            mu=config["momentum"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    steps_per_epoch = int(np.ceil(train_labels.shape[0] / BATCH_SIZE))
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "learning_rate": [],
    }

    best_valid_acc = -1.0
    best_model_path = weights_dir / f"{config['name']}_best_model.pickle"
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
            if scheduler is not None:
                scheduler.step()

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
        history["learning_rate"].append(float(optimizer.init_lr))

        print(
            f"{config['name']} epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}, "
            f"lr={optimizer.init_lr:.5f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            model.save_model(best_model_path)

    best_model = nn.models.Model_CNN()
    best_model.load_model(best_model_path)
    best_loss_fn = nn.op.MultiCrossEntropyLoss(model=best_model, max_classes=10)
    test_loss, test_acc = evaluate(best_model, best_loss_fn, test_images, test_labels)

    summary = copy.deepcopy(config)
    summary.update(
        {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
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
            "generalization_gap": float(history["train_acc"][-1] - history["valid_acc"][-1]),
            "best_model_path": str(best_model_path),
        }
    )

    with open(output_dir / f"{config['name']}_history.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "history": history}, f, indent=2)

    return summary, history


def plot_group(results, category, output_path):
    selected = [(name, data) for name, data in results.items() if data["summary"]["category"] == category]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name, data in selected:
        history = data["history"]
        label = name.replace("cnn_", "")
        axes[0].plot(history["epoch"], history["valid_loss"], marker="o", label=label)
        axes[1].plot(history["epoch"], history["valid_acc"], marker="o", label=label)

    axes[0].set_title(f"{category}: validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title(f"{category}: validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_summary(results, output_path):
    names = ["partb_sgd"] + [name.replace("cnn_", "") for name in results.keys()]
    valid_acc = [PART_B_REFERENCE["valid_acc"]] + [data["summary"]["best_valid_acc"] for data in results.values()]
    test_acc = [PART_B_REFERENCE["test_acc"]] + [data["summary"]["test_acc"] for data in results.values()]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width / 2, valid_acc, width, label="Validation")
    ax.bar(x + width / 2, test_acc, width, label="Test")
    ax.axhline(PART_B_REFERENCE["test_acc"], color="gray", linestyle="--", linewidth=1, label="Part B test")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.90, 0.98)
    ax.set_ylabel("Accuracy")
    ax.set_title("Part C experiment comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    np.random.seed(SEED)

    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent
    output_dir = project_root / "PartC"
    weights_dir = project_root / "model_weights_for_upload" / "PartC"
    output_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    data_dir = code_dir / "dataset" / "MNIST"
    train_images = load_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    split_path = project_root / "PartA" / "parta_split_indices.npy"
    indices = np.load(split_path)
    train_images = train_images[indices]
    train_labels = train_labels[indices]
    valid_images = train_images[:10000]
    valid_labels = train_labels[:10000]
    train_images = train_images[10000:]
    train_labels = train_labels[10000:]

    datasets = (train_images, train_labels, valid_images, valid_labels, test_images, test_labels)
    results = {}

    for config in EXPERIMENTS:
        print(f"\n=== Running {config['name']} ===")
        summary, history = train_one(config, datasets, output_dir, weights_dir)
        results[config["name"]] = {"summary": summary, "history": history}

    plot_group(results, "Optimization", output_dir / "optimization_curves.png")
    plot_group(results, "Regularization", output_dir / "regularization_curves.png")
    plot_summary(results, output_dir / "partc_summary.png")

    final_summary = {
        "seed": SEED,
        "reference": PART_B_REFERENCE,
        "experiments": {name: data["summary"] for name, data in results.items()},
    }
    with open(output_dir / "partc_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": final_summary, "results": results}, f, indent=2)

    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
