from pathlib import Path

import torch
from model import TinyGPT


@torch.no_grad()
def generate(model, token_ids, max_new_tokens, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    model.eval()

    for _ in range(max_new_tokens):
        # Keep only the most recent characters the model can process.
        context = token_ids[:, -model.block_size:]
        logits = model(context)

        # Use the final position to predict the next character.
        next_logits = logits[:, -1, :] / temperature
        probabilities = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        # Append the sampled character ID.
        token_ids = torch.cat([token_ids, next_id], dim=1)

    return token_ids


def main():
    checkpoint_path = Path(__file__).parent / "nature_gpt.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "nature_gpt.pt is missing. Run inspect_data.py to train and save it."
        )

    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model = TinyGPT(**saved["config"])
    model.load_state_dict(saved["model_state"])
    model.eval()

    # Use the exact vocabulary saved during training.
    chars = saved["chars"]
    stoi = {
        character: index
        for index, character in enumerate(chars)
    }

    prompt = input("Starting text: ")

    if not prompt:
        raise ValueError("Enter at least one character.")

    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(
            f"Characters outside the vocabulary: {unknown!r}"
        )

    prompt_ids = torch.tensor(
        [[stoi[character] for character in prompt]],
        dtype=torch.long
    )

    torch.manual_seed(42)

    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=300,
        temperature=0.7
    )

    generated_text = "".join(
        chars[index] for index in generated_ids[0].tolist()
    )
    
    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()