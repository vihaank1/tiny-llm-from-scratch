# Tiny GPT from Scratch

An educational character-level language model built with PyTorch.
Attention and Transformer blocks are implemented directly, without
prebuilt Transformer layers or pretrained model weights.

## Architecture

- Character tokenization
- Learned token and position embeddings
- Causal scaled dot-product attention
- 4 attention heads per block
- 4 Transformer blocks
- 128-dimensional embeddings
- 128-character context window
- Feed-forward layers, residual connections, and layer normalization
- Dropout of 0.2 on residual branches

## Training

Uses AdamW and next-character cross-entropy loss.
Validation runs every 500 updates. Training stops after four
evaluations without improvement and restores the best checkpoint.

The corpus contains nature writing by John Muir:
- My First Summer in the Sierra — https://www.gutenberg.org/ebooks/32540
- Travels in Alaska — https://www.gutenberg.org/ebooks/7345
- The Mountains of California — https://www.gutenberg.org/ebooks/10012

The original held-out validation passage remains separate.
Added paragraphs sharing an exact 20-word sequence with validation
were excluded.

## Run

Install dependencies:

    python -m pip install -r requirements.txt

Train:

    python -u inspect_data.py

Generate from the saved checkpoint:

    python generate.py

Training starts a fresh model. Generation loads nature_gpt.pt.
Checkpoints are stored locally and excluded from Git.

## Results and limitations

Best observed validation cross-entropy: 1.2459 nats per character.
Checkpoint reload and vocabulary checks passed.

The model generates recognizable words and nature-related phrases,
but grammar, spelling, repetition, and paragraph coherence remain
unreliable. It is an educational implementation, not a reliable
writing assistant or factual information source.