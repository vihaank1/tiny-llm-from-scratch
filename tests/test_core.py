"""Small regression tests for causality, sampling, and checkpoint compatibility."""
import unittest
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluate import baseline_scores
from generate import generate
from model import TinyGPT, TransformerBlock
from train import get_batch, estimate_loss


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_full_model_causality(self):
        model = TinyGPT(9, 8, 16, 2, 2).eval()
        x = torch.randint(9, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 9
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x)[:, :4], model(changed)[:, :4]))

    def test_dropout_is_called_twice_and_branches_once(self):
        block = TransformerBlock(16, 2, dropout=0.2)
        calls = {name: 0 for name in ("attention", "feed_forward", "dropout")}
        hooks = []
        for name in calls:
            def count(_module, _inputs, _output, name=name):
                calls[name] += 1
            hooks.append(getattr(block, name).register_forward_hook(count))
        x = torch.randn(2, 8, 16)
        block(x).square().mean().backward()
        self.assertEqual(calls, {"attention": 1, "feed_forward": 1, "dropout": 2})
        grad = block.attention.heads[0].query.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)
        for hook in hooks:
            hook.remove()

    def test_old_config_compatible_and_eval_dropout_identity(self):
        config = dict(vocab_size=9, block_size=8, embedding_size=16, num_heads=2, num_layers=2)
        model = TinyGPT(**config).eval()
        no_dropout = TinyGPT(**config, dropout=0).eval()
        no_dropout.load_state_dict(model.state_dict(), strict=True)
        x = torch.randint(9, (2, 8))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), no_dropout(x)))

    def test_generation_crops_context_and_top_one_is_seed_independent(self):
        model = TinyGPT(9, 8, 16, 2, 1)
        prompt = torch.ones((2, 12), dtype=torch.long)
        a = generate(model, prompt, 10, top_k=1)
        torch.manual_seed(99)
        b = generate(model, prompt, 10, top_k=1)
        self.assertEqual(a.shape, (2, 22))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, :12], prompt))
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            generate(model, prompt, 1, temperature=0)

    def test_shifted_batches_and_repeatable_evaluation(self):
        source = torch.arange(100) % 9
        x, y = get_batch(source, 8, 4)
        self.assertEqual(x.shape, (4, 8))
        self.assertTrue(torch.equal((x + 1) % 9, y))
        model = TinyGPT(9, 8, 16, 2, 1)
        args = (model, {"train": source, "val": source}, 8, 4, 2, 100)
        self.assertEqual(estimate_loss(*args), estimate_loss(*args))
        self.assertTrue(model.training)

    def test_baseline_on_alternating_sequence(self):
        ids = np.array([0, 1] * 100)
        scores = baseline_scores(ids, ids[:20], 2)
        self.assertLess(scores["bigram"], scores["unigram"])
        self.assertAlmostEqual(scores["uniform"], scores["unigram"])

    def test_early_stopping_counts_evaluations_not_updates(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            corpus = folder / "text.txt"
            corpus.write_text("The forest and the trees. " * 30)
            run = folder / "run"
            command = [sys.executable, str(root / "train.py"),
                       "--train-data", str(corpus), "--val-data", str(corpus),
                       "--run-dir", str(run), "--checkpoint", str(folder / "model.pt"),
                       "--max-steps", "8", "--eval-every", "2", "--patience", "2",
                       "--eval-batches", "1", "--batch-size", "2", "--block-size", "8",
                       "--embedding-size", "8", "--num-heads", "2", "--num-layers", "1",
                       "--sample-length", "0", "--threads", "1",
                       "--learning-rate", "1e-30", "--weight-decay", "0"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertEqual(metadata["completed_steps"], 4)
            self.assertEqual(metadata["best_step"], 0)
            self.assertEqual(metadata["stop_reason"], "early_stopping")
            self.assertTrue(metadata["reload_matches"])
            with (run / "metrics.csv").open() as handle:
                self.assertEqual([int(row["step"]) for row in csv.DictReader(handle)], [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
