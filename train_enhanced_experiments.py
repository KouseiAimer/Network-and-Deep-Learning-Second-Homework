import gzip
import json
import shutil
from pathlib import Path
from struct import unpack

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


SEED = 309
BATCH_SIZE = 256

REFERENCE = {
    "mlp_test_acc": 0.9318,
    "cnn_partb_test_acc": 0.9605,
    "momentum_partc_test_acc": 0.9706,
}


EXPERIMENTS = [
    {
        "name": "enhanced_deep_cnn_adam",
        "category": "Enhanced CNN",
        "model_type": "deep",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "dropout_rate": 0.1,
        "epochs": 8,
        "early_stopping": None,
        "description": "Two-convolution CNN with Adam and dropout=0.1",
    },
    {
        "name": "cnn_adam",
        "category": "Optimization",
        "model_type": "single",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "dropout_rate": 0.0,
        "epochs": 5,
        "early_stopping": None,
        "description": "Single-convolution CNN with Adam",
    },
    {
        "name": "cnn_rmsprop",
        "category": "Optimization",
        "model_type": "single",
        "optimizer": "rmsprop",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "dropout_rate": 0.0,
        "epochs": 5,
        "early_stopping": None,
        "description": "Single-convolution CNN with RMSProp",
    },
    {
        "name": "cnn_dropout_momentum",
        "category": "Regularization",
        "model_type": "single",
        "optimizer": "momentum",
        "learning_rate": 0.02,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "dropout_rate": 0.2,
        "epochs": 5,
        "early_stopping": None,
        "description": "Momentum CNN with dropout=0.2",
    },
    {
        "name": "cnn_early_stopping",
        "category": "Regularization",
        "model_type": "single",
        "optimizer": "momentum",
        "learning_rate": 0.02,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "dropout_rate": 0.0,
        "epochs": 12,
        "early_stopping": {"patience": 2, "min_delta": 1e-4},
        "description": "Momentum CNN with validation-based early stopping",
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


def set_training(model, training):
    if hasattr(model, "set_training"):
        model.set_training(training)


def initialize_he(model):
    for layer in model.layers:
        if layer.optimizable:
            if layer.W.ndim == 4:
                fan_in = layer.W.shape[1] * layer.W.shape[2] * layer.W.shape[3]
            else:
                fan_in = layer.W.shape[0]
            layer.W[:] = np.random.normal(0.0, np.sqrt(2.0 / fan_in), size=layer.W.shape)
            layer.b[:] = 0.0


def build_model(config):
    common = {
        "weight_decay": config["weight_decay"] > 0,
        "weight_decay_lambda": max(config["weight_decay"], 1e-12),
        "dropout_rate": config["dropout_rate"],
    }
    if config["model_type"] == "single":
        model = nn.models.Model_CNN(
            conv_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_dim=64,
            **common,
        )
    elif config["model_type"] == "deep":
        model = nn.models.Model_CNN_Deep(
            conv1_channels=8,
            conv2_channels=16,
            kernel_size=3,
            stride1=1,
            stride2=2,
            padding=1,
            hidden_dim=128,
            **common,
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    initialize_he(model)
    return model


def build_optimizer(config, model):
    if config["optimizer"] == "sgd":
        return nn.optimizer.SGD(init_lr=config["learning_rate"], model=model)
    if config["optimizer"] == "momentum":
        return nn.optimizer.MomentGD(
            init_lr=config["learning_rate"],
            model=model,
            mu=config.get("momentum", 0.9),
        )
    if config["optimizer"] == "adam":
        return nn.optimizer.Adam(init_lr=config["learning_rate"], model=model)
    if config["optimizer"] == "rmsprop":
        return nn.optimizer.RMSProp(init_lr=config["learning_rate"], model=model)
    raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def evaluate(model, loss_fn, images, labels, batch_size=1024):
    set_training(model, False)
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

    set_training(model, True)
    return total_loss / total, total_correct / total


def train_one(config, datasets, output_dir, weights_dir):
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = datasets
    model = build_model(config)
    optimizer = build_optimizer(config, model)
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
    best_epoch = 0
    best_model_path = weights_dir / f"{config['name']}_best_model.pickle"
    num_train = train_labels.shape[0]
    patience_counter = 0
    stopped_early = False

    for epoch in range(1, config["epochs"] + 1):
        set_training(model, True)
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
        history["learning_rate"].append(float(optimizer.init_lr))

        print(
            f"{config['name']} epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

        min_delta = 0.0
        if config["early_stopping"] is not None:
            min_delta = config["early_stopping"].get("min_delta", 0.0)

        if valid_acc > best_valid_acc + min_delta:
            best_valid_acc = valid_acc
            best_epoch = epoch
            patience_counter = 0
            model.save_model(best_model_path)
        else:
            patience_counter += 1

        if config["early_stopping"] is not None:
            patience = config["early_stopping"]["patience"]
            if patience_counter >= patience:
                stopped_early = True
                print(f"{config['name']} stopped early at epoch {epoch}")
                break

    if config["model_type"] == "deep":
        best_model = nn.models.Model_CNN_Deep()
    else:
        best_model = nn.models.Model_CNN()
    best_model.load_model(best_model_path)
    best_loss_fn = nn.op.MultiCrossEntropyLoss(model=best_model, max_classes=10)
    test_loss, test_acc = evaluate(best_model, best_loss_fn, test_images, test_labels)

    summary = {
        **config,
        "batch_size": BATCH_SIZE,
        "epochs_ran": len(history["epoch"]),
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_valid_loss": history["valid_loss"][-1],
        "final_valid_acc": history["valid_acc"][-1],
        "best_valid_acc": float(best_valid_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "stopped_early": stopped_early,
        "best_model_path": str(best_model_path),
    }

    with open(output_dir / f"{config['name']}_history.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "history": history}, f, indent=2)

    return summary, history


def plot_curves(results, selected_names, output_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in selected_names:
        history = results[name]["history"]
        label = name.replace("cnn_", "").replace("enhanced_", "")
        axes[0].plot(history["epoch"], history["valid_loss"], marker="o", label=label)
        axes[1].plot(history["epoch"], history["valid_acc"], marker="o", label=label)

    axes[0].set_title(f"{title}: validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title(f"{title}: validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_summary(results, output_path):
    names = ["PartB CNN", "PartC Momentum"] + [name.replace("cnn_", "").replace("enhanced_", "") for name in results]
    tests = [REFERENCE["cnn_partb_test_acc"], REFERENCE["momentum_partc_test_acc"]] + [
        results[name]["summary"]["test_acc"] for name in results
    ]
    valids = [0.9585, 0.9716] + [results[name]["summary"]["best_valid_acc"] for name in results]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - width / 2, valids, width, label="Validation")
    ax.bar(x + width / 2, tests, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.92, 0.99)
    ax.set_ylabel("Accuracy")
    ax.set_title("Extended experiment comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def copy_existing_weights(project_root, weights_dir):
    sources = [
        project_root / "PartA" / "mlp_best_model.pickle",
        project_root / "model_weights_for_upload" / "partb_cnn_best_model.pickle",
    ]
    partc_old = project_root / "model_weights_for_upload" / "PartC"
    if partc_old.exists():
        sources.extend(sorted(partc_old.glob("*.pickle")))
    for source in sources:
        if source.exists():
            target = weights_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)


def main():
    np.random.seed(SEED)
    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent
    output_dir = project_root / "PartC" / "extended"
    weights_dir = project_root / "model_weight"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    copy_existing_weights(project_root, weights_dir)

    data_dir = code_dir / "dataset" / "MNIST"
    train_images = load_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    indices = np.load(project_root / "PartA" / "parta_split_indices.npy")
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

    plot_curves(
        results,
        ["cnn_adam", "cnn_rmsprop"],
        output_dir / "extra_optimization_curves.png",
        "Extra optimizers",
    )
    plot_curves(
        results,
        ["cnn_dropout_momentum", "cnn_early_stopping"],
        output_dir / "extra_regularization_curves.png",
        "Extra regularization",
    )
    plot_curves(
        results,
        ["enhanced_deep_cnn_adam"],
        output_dir / "enhanced_cnn_curve.png",
        "Enhanced CNN",
    )
    plot_summary(results, output_dir / "extended_summary.png")

    final = {"seed": SEED, "reference": REFERENCE, "experiments": results}
    with open(output_dir / "extended_results.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print(json.dumps({k: v["summary"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
