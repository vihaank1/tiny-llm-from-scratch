"""Generate characters from a saved TinyGPT checkpoint; importing never trains."""
import argparse
from pathlib import Path

import torch

from model import TinyGPT

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def generate(model, token_ids, max_new_tokens, temperature=1.0, top_k=None):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if token_ids.ndim != 2 or token_ids.shape[1] == 0:
        raise ValueError("token_ids must have shape [batch, nonempty sequence]")

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = token_ids[:, -model.block_size:]
            # [B, T, V] -> [B, V]: scores for the next character only.
            next_logits = model(context)[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, next_logits.shape[-1])
                values, indices = torch.topk(next_logits, k, dim=-1)
                filtered = torch.full_like(next_logits, float("-inf"))
                next_logits = filtered.scatter(-1, indices, values)
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat([token_ids, next_id], dim=1)
    finally:
        model.train(was_training)
    return token_ids


def encode_prompt(prompt, chars):
    if not prompt:
        raise ValueError("Enter at least one character.")
    stoi = {character: index for index, character in enumerate(chars)}
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"Characters outside the vocabulary: {unknown!r}")
    return torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "nature_gpt.pt")
    parser.add_argument("--prompt", help="Prompt; omitted means interactive input")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint missing. Train with train.py or specify --checkpoint.")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyGPT(**saved["config"])
    model.load_state_dict(saved["model_state"])
    model.eval()
    prompt = args.prompt if args.prompt is not None else input("Starting text: ")
    try:
        ids = encode_prompt(prompt, saved["chars"])
        torch.manual_seed(args.seed)
        result = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    print("\nGenerated text:\n")
    print("".join(saved["chars"][i] for i in result[0].tolist()))


if __name__ == "__main__":
    main()
