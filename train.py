"""Train the educational character GPT, recording metrics and reproducibility data."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import encode_prompt, generate
from model import TinyGPT

ROOT = Path(__file__).resolve().parent


def get_batch(source, block_size, batch_size, generator=None):
    if len(source) <= block_size:
        raise ValueError("Each split needs at least block_size + 1 characters")
    starts = torch.randint(len(source) - block_size, (batch_size,), generator=generator)
    inputs = torch.stack([source[i:i + block_size] for i in starts.tolist()])
    targets = torch.stack([source[i + 1:i + block_size + 1] for i in starts.tolist()])
    return inputs, targets


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, batches, seed):
    was_training = model.training
    model.eval()
    results = {}
    try:
        for offset, (name, source) in enumerate(splits.items()):
            # The same windows at each evaluation make early stopping less noisy.
            generator = torch.Generator().manual_seed(seed + offset)
            losses = []
            for _ in range(batches):
                x, y = get_batch(source, block_size, batch_size, generator)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                losses.append(loss.item())
            results[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/train.txt")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data/val_fixed.txt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "nature_gpt_v2.pt")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/latest")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--time-limit", type=float, default=900)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--prompt", default="The forest")
    parser.add_argument("--sample-length", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    positive = [args.max_steps, args.time_limit, args.eval_every, args.eval_batches,
                args.patience, args.batch_size, args.block_size, args.embedding_size,
                args.num_heads, args.num_layers, args.learning_rate]
    if any(value <= 0 for value in positive) or not 0 <= args.dropout < 1:
        parser.error("Training sizes/rates must be positive; dropout must be in [0, 1).")
    if args.sample_length < 0 or args.weight_decay < 0:
        parser.error("Sample length and weight decay must be nonnegative.")
    if args.embedding_size % args.num_heads:
        parser.error("embedding-size must be divisible by num-heads")
    if not args.overwrite and (args.checkpoint.exists() or (args.run_dir / "metadata.json").exists()):
        parser.error("Output already exists. Choose new paths or explicitly use --overwrite.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    train_text = args.train_data.read_text(encoding="utf-8")
    val_text = args.val_data.read_text(encoding="utf-8")
    # Keep the original project's vocabulary policy for compatibility.
    chars = sorted(set(train_text + val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    splits = {name: torch.tensor([stoi[c] for c in text], dtype=torch.long)
              for name, text in [("train", train_text), ("val", val_text)]}
    if min(map(len, splits.values())) <= args.block_size:
        parser.error("Both splits need more characters than block-size")
    try:
        prompt_ids = encode_prompt(args.prompt, chars)
    except ValueError as exc:
        parser.error(str(exc))
    config = dict(vocab_size=len(chars), block_size=args.block_size,
                  embedding_size=args.embedding_size, num_heads=args.num_heads,
                  num_layers=args.num_layers, dropout=args.dropout)
    model = TinyGPT(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    metadata = {
        "config": config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(), "pytorch": str(torch.__version__),
        "system": platform.system(), "machine": platform.machine(),
        "processor": platform.processor() or "not reported by platform",
        "device": "cpu", "threads": torch.get_num_threads(), "seed": args.seed,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_steps": args.max_steps,
        "eval_every": args.eval_every, "eval_batches": args.eval_batches,
        "eval_seed": args.seed + 1000, "patience": args.patience,
        "time_limit_seconds": args.time_limit,
        "train_characters": len(train_text), "val_characters": len(val_text),
        "train_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
        "val_sha256": hashlib.sha256(val_text.encode()).hexdigest(),
        "vocabulary_policy": "sorted union of training and validation characters",
        "sampling": {"prompt": args.prompt, "temperature": 0.7, "seed": 42,
                     "top_k": None, "new_characters": args.sample_length},
        "evaluation": "fixed random windows, mean next-character cross-entropy",
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    best_val, best_step, stale = float("inf"), 0, 0
    probe_ids = prompt_ids[:, -args.block_size:]
    best_probe_logits = None
    completed_steps = 0
    last_evaluated = -1
    start = time.perf_counter()
    stop_reason = "max_steps"

    with (args.run_dir / "metrics.csv").open("w", newline="") as metric_file, \
         (args.run_dir / "samples.jsonl").open("w", encoding="utf-8") as samples:
        writer = csv.DictWriter(metric_file, fieldnames=["step", "train", "val", "elapsed"])
        writer.writeheader()

        def evaluate_and_save(step):
            nonlocal best_val, best_step, stale, last_evaluated, best_probe_logits
            results = estimate_loss(model, splits, args.block_size, args.batch_size,
                                    args.eval_batches, args.seed + 1000)
            row = dict(step=step, **results, elapsed=time.perf_counter() - start)
            writer.writerow(row)
            metric_file.flush()
            last_evaluated = step
            print(f"Step {step}: train={results['train']:.4f}, val={results['val']:.4f}, "
                  f"elapsed={row['elapsed']:.1f}s", flush=True)
            if results["val"] < best_val:
                best_val, best_step, stale = results["val"], step, 0
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    best_probe_logits = model(probe_ids).clone()
                model.train(was_training)
                torch.save({"model_state": model.state_dict(), "config": config,
                            "chars": chars, "best_step": best_step,
                            "best_val_loss": best_val, "metadata": metadata}, args.checkpoint)
                print(f"New best validation loss: {best_val:.4f}", flush=True)
            else:
                stale += 1
            # Record real same-prompt samples without altering the training RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                ids = generate(model, prompt_ids, args.sample_length, temperature=0.7)
            samples.write(json.dumps({"step": step, "text": "".join(chars[i] for i in ids[0].tolist()),
                                      **metadata["sampling"]}, ensure_ascii=False) + "\n")
            samples.flush()

        model.train()
        evaluate_and_save(0)
        while completed_steps < args.max_steps:
            if stale >= args.patience:
                stop_reason = "early_stopping"
                break
            if time.perf_counter() - start >= args.time_limit:
                stop_reason = "time_limit"
                break
            x, y = get_batch(splits["train"], args.block_size, args.batch_size)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            completed_steps += 1
            if completed_steps % args.eval_every == 0:
                evaluate_and_save(completed_steps)
        # Evaluate a final partial interval once, including when the time cap fires.
        if last_evaluated != completed_steps:
            evaluate_and_save(completed_steps)

    elapsed = time.perf_counter() - start
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"])
    model.eval()
    reloaded = TinyGPT(**saved["config"])
    reloaded.load_state_dict(saved["model_state"])
    reloaded.eval()
    with torch.no_grad():
        reload_matches = torch.allclose(best_probe_logits, reloaded(probe_ids))
    metadata.update(completed_steps=completed_steps, best_step=best_step,
                    best_val_loss=best_val, best_val_bpc=best_val / math.log(2),
                    elapsed_seconds=elapsed, stop_reason=stop_reason,
                    reload_matches=reload_matches, vocabulary_matches=saved["chars"] == chars)
    saved["metadata"] = metadata
    torch.save(saved, args.checkpoint)
    (args.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {completed_steps} updates in {elapsed:.1f} seconds")
    print(f"Restored best validation loss: {best_val:.4f} (step {best_step})")
    print(f"Reload matches: {reload_matches}; vocabulary matches: {saved['chars'] == chars}")
    print(f"Saved {args.checkpoint.name}")
    print("Generate with: python generate.py --checkpoint", str(args.checkpoint))


if __name__ == "__main__":
    main()
