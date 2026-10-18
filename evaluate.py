"""Measure corpus statistics and baselines, optionally evaluating a real checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def baseline_scores(train_ids, val_ids, vocab_size, alpha=1.0):
    """Add-alpha counts learned only from training; score val[1:] for all models."""
    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.float64)
    unigram = (counts + alpha) / (counts.sum() + alpha * vocab_size)
    pairs = np.bincount(train_ids[:-1] * vocab_size + train_ids[1:],
                        minlength=vocab_size ** 2).reshape(vocab_size, vocab_size)
    bigram = (pairs + alpha) / (pairs.sum(axis=1, keepdims=True) + alpha * vocab_size)
    targets = val_ids[1:]
    return {
        "uniform": math.log(vocab_size),
        "unigram": float(-np.log(unigram[targets]).mean()),
        "bigram": float(-np.log(bigram[val_ids[:-1], targets]).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/train.txt")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data/val_fixed.txt")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/baselines.json")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    train = args.train_data.read_text(encoding="utf-8")
    val = args.val_data.read_text(encoding="utf-8")
    if len(train) < 2 or len(val) < 2:
        parser.error("Each split must contain at least two characters")
    chars = sorted(set(train + val))
    saved = None
    if args.checkpoint:
        import torch
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        chars = saved["chars"]
        if set(train + val) - set(chars):
            parser.error("Checkpoint vocabulary does not cover the supplied corpus")
    stoi = {c: i for i, c in enumerate(chars)}
    train_ids = np.array([stoi[c] for c in train], dtype=np.int64)
    val_ids = np.array([stoi[c] for c in val], dtype=np.int64)
    scores = baseline_scores(train_ids, val_ids, len(chars), args.alpha)
    results = {
        "train_characters": len(train), "train_words_whitespace": len(train.split()),
        "validation_characters": len(val), "validation_words_whitespace": len(val.split()),
        "vocab_size": len(chars), "validation_targets": len(val) - 1,
        "train_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "validation_sha256": hashlib.sha256(val.encode()).hexdigest(),
        "alpha": args.alpha, "training_counts_only": True,
        "evaluation": "All validation next-character targets, excluding first character",
    }
    if saved:
        import torch.nn.functional as F
        from model import TinyGPT
        model = TinyGPT(**saved["config"])
        model.load_state_dict(saved["model_state"])
        model.eval()
        data = torch.tensor(val_ids, dtype=torch.long)
        total_loss, total_targets = 0.0, 0
        # Each target is scored exactly once. Context resets every block_size positions.
        with torch.no_grad():
            for start in range(0, len(data) - 1, model.block_size):
                end = min(start + model.block_size, len(data) - 1)
                x = data[start:end].unsqueeze(0)
                y = data[start + 1:end + 1]
                logits = model(x).reshape(-1, len(chars))
                total_loss += F.cross_entropy(logits, y, reduction="sum").item()
                total_targets += len(y)
        scores["gpt"] = total_loss / total_targets
        results["parameter_count"] = sum(p.numel() for p in model.parameters())
        results["checkpoint_config"] = saved["config"]
        results["context_policy"] = "nonoverlapping blocks; context resets at each block"
        results["bigram_minus_gpt_nats"] = scores["bigram"] - scores["gpt"]
    results["scores"] = {name: {"nats_per_character": loss, "bits_per_character": loss / math.log(2)}
                         for name, loss in scores.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
