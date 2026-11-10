import csv
import gzip
import json
import random
import struct
import time
import argparse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path.home() / "mnist-project"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.01
HIDDEN_SIZE = 128
NUM_THREADS = 1

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(BASE_URL + filename, path)


def load_images(filename):
    with gzip.open(DATA_DIR / filename, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file format: {filename}")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(count, rows * cols).copy()

    # Match the agreed preprocessing: float32 pixels in [0, 1].
    return torch.from_numpy(images).to(torch.float32).div_(255.0)


def load_labels(filename):
    with gzip.open(DATA_DIR / filename, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file format: {filename}")
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()

    if len(labels) != count:
        raise ValueError(f"Unexpected label count: {filename}")

    return torch.from_numpy(labels.astype(np.int64))


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intra-threads",
        type=int,
        choices=[1, 2, 4, 8],
        default=1
    )
    args = parser.parse_args()    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    torch.set_num_threads(args.intra_threads)
    torch.set_num_interop_threads(1)
    print("Loading MNIST...")
    download_files()

    x_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    x_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")

    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )

    model = nn.Sequential(
        nn.Linear(784, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 10),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE
    )

    print("Framework:", torch.__version__)
    print("Device: CPU")
    print("Intra-op threads:", torch.get_num_threads())
    print("Inter-op threads:", torch.get_num_interop_threads())
    print("Training:", len(x_train), "images")
    print("Testing:", len(x_test), "images")
    print("Starting training...")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    epoch_rows = []

    # Model creation and data preparation are excluded from this timer.
    train_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0

        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += n

        avg_loss = loss_sum / seen
        accuracy = 100.0 * correct / seen

        epoch_rows.append({
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_accuracy_percent": accuracy,
        })

        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"loss={avg_loss:.4f}, accuracy={accuracy:.2f}%"
        )

    training_seconds = time.perf_counter() - train_start

    # Warm-up is not included in the inference measurement.
    model.eval()
    with torch.inference_mode():
        _ = model(x_test[:BATCH_SIZE])

        inference_start = time.perf_counter()
        logits_batches = []

        for start in range(0, len(x_test), BATCH_SIZE):
            batch = x_test[start:start + BATCH_SIZE]
            logits_batches.append(model(batch))

        inference_seconds = time.perf_counter() - inference_start

        all_logits = torch.cat(logits_batches)
        test_loss = loss_fn(all_logits, y_test).item()
        correct = (all_logits.argmax(dim=1) == y_test).sum().item()

    test_accuracy = 100.0 * correct / len(y_test)

    train_images_processed = len(x_train) * EPOCHS
    train_throughput = train_images_processed / training_seconds
    inference_throughput = len(x_test) / inference_seconds

    summary = {
        "run_id": run_id,
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "device": "CPU",
        "seed": SEED,
        "architecture": f"784-{HIDDEN_SIZE}-10",
        "activation": "ReLU",
        "optimizer": "SGD",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "inference_batch_size": BATCH_SIZE,
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "train_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "test_loss": test_loss,
        "test_accuracy_percent": test_accuracy,
        "train_images_processed": train_images_processed,
        "train_images_per_second": train_throughput,
        "test_images_processed": len(x_test),
        "inference_images_per_second": inference_throughput,
    }

    append_csv(RESULTS_DIR / "epoch_metrics.csv", epoch_rows[0])
    for row in epoch_rows[1:]:
        append_csv(RESULTS_DIR / "epoch_metrics.csv", row)

    append_csv(RESULTS_DIR / "run_summary.csv", summary)

    with (RESULTS_DIR / f"config_{run_id}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BASELINE RESULTS ===")
    print(f"Training time: {training_seconds:.3f} seconds")
    print(f"Inference time: {inference_seconds:.3f} seconds")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Training throughput: {train_throughput:.1f} images/sec")
    print(f"Inference throughput: {inference_throughput:.1f} images/sec")
    print("\nResults saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
