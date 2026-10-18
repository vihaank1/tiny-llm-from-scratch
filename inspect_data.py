import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy

with open("data/train.txt", "r", encoding="utf-8") as f:
    train_text = f.read()

with open("data/val_fixed.txt", "r", encoding="utf-8") as f:
    val_text = f.read()

chars = sorted(set(train_text + val_text))
vocab_size = len(chars)
stoi = {character: index for index, character in enumerate(chars)}

train_data = torch.tensor(
    [stoi[character] for character in train_text],
    dtype=torch.long,
)

val_data = torch.tensor(
    [stoi[character] for character in val_text],
    dtype=torch.long,
)

# Used by the existing diagnostic prints.
data = torch.cat([train_data, val_data])

print("Vocabulary size:", vocab_size)
print("Training shape:", train_data.shape)
print("Validation shape:", val_data.shape)

block_size = 128
batch_size = 16


def get_batch(split):
    if split == "train":
        source = train_data
    elif split == "val":
        source = val_data
    else:
        raise ValueError("split must be 'train' or 'val'")

    starts = torch.randint(
        0, len(source) - block_size, (batch_size,)
    ).tolist()

    inputs = []
    targets = []

    for start in starts:
        inputs.append(source[start:start + block_size])
        targets.append(source[start + 1:start + block_size + 1])

    return torch.stack(inputs), torch.stack(targets)


print("Tensor shape:", data.shape)
print("Tensor dtype:", data.dtype)

for split in ["train", "val"]:
    xc, yc = get_batch(split)
    print(split, xc.shape, yc.shape)
    print("Overlap:", torch.equal(xc[:, 1:], yc[:, :-1]))


embedding_size=128
token_embedding=nn.Embedding(vocab_size, embedding_size)

yb, zb = get_batch("train")
embedded = token_embedding(yb)

print("Embedding table:", token_embedding.weight.shape)
print("Input IDs:", yb.shape)
print("Embedded input:", embedded.shape)


position_embedding=nn.Embedding(block_size, embedding_size)

positions=torch.arange(yb.shape[1])
position_vectors=position_embedding(positions)

x=embedded+position_vectors

print("Position IDs:", positions)
print("Position vectors:", position_vectors.shape)
print("Combined embeddings:", x.shape)


head_size=8

query=nn.Linear(embedding_size, head_size, bias=False)
key=nn.Linear(embedding_size, head_size, bias=False)
value=nn.Linear(embedding_size, head_size, bias=False)

q=query(x)
k=key(x)
v=value(x)

print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)


scores = (q @ k.transpose(-2,-1)) / (head_size ** 0.5)

print("Score shape:", scores.shape)
print("First sequence scores:", scores[0])

sequence_length=x.shape[1]

mask=torch.tril(
    torch.ones(
        sequence_length, sequence_length,
        dtype=torch.bool, device=x.device
    )
)

masked_scores=scores.masked_fill(~mask, float("-inf"))
weights=torch.softmax(masked_scores, dim=-1)

print("Weights:", weights[0])
print("Row totals:", weights[0].sum(dim=-1))


attention_output = weights @ v

print("Attention output shape:", attention_output.shape)
print(
    "First position matches its value:",
    torch.allclose(attention_output[:,0,:], v[:,0,:])
)


from model import AttentionHead, MultiHeadAttention

head=AttentionHead(embedding_size, head_size)

with torch.no_grad():
    output=head(x)

    changed_x=x.clone()
    changed_x[:, 4:, :] += 10
    changed_output=head(changed_x)

print("Head output:", output.shape)
print(
    "Earlier positions unaffected:",
    torch.allclose(output[:, :4, :], changed_output[:, :4, :])
)

multi_head=MultiHeadAttention(embedding_size, num_heads=2)

with torch.no_grad():
    output=multi_head(x)

    changed_x=x.clone()
    changed_x[:, 4:, :] += 10
    changed_output=multi_head(changed_x)

print("Multi-head output:", output.shape)
print(
    "Earlier positions unaffected:",
    torch.allclose(output[:, :4, :], changed_output[:, :4, :])
)


from model import FeedForward, TransformerBlock

feed_forward=FeedForward(embedding_size)
ff_output=feed_forward(x)

print("Feed-forward output:", ff_output.shape)

block=TransformerBlock(embedding_size, num_heads=2)
block.eval()

with torch.no_grad():
    block_output=block(x)

    changed_x=x.clone()
    changed_x[:, 4: :] += torch.randn_like(changed_x[:, 4:, :])
    changed_output=block(changed_x)

print("Block output:", block_output.shape)
print(
    "Block remains causal:",
    torch.allclose(block_output[:, :4, :], changed_output[:, :4, :])
)


from model import TinyGPT

model = TinyGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_size=embedding_size,
    num_heads=4,
    num_layers=4
)
xb,yb=get_batch("train")

logits=model(xb)

print("Logits shape:", logits.shape)
print("Target shape:", yb.shape)
print("Trainable parameters:", sum(
    p.numel() for p in model.parameters() if p.requires_grad
))

loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    yb.reshape(-1)
)

print("Initial loss:", loss.item())

optimizer=torch.optim.AdamW(model.parameters(), lr=.001)

optimizer.zero_grad(set_to_none=True)
loss.backward()
attention_grad=model.blocks[0].attention.heads[0].query.weight.grad
print("Attention receives gradients:", attention_grad is not None)
optimizer.step()

with torch.no_grad():
    updated_logits=model(xb)
    updated_loss=F.cross_entropy(
        updated_logits.reshape(-1,vocab_size),
        yb.reshape(-1)
    )

print("Same-batch loss after 1 update:", updated_loss.item())

@torch.no_grad()
def estimate_loss():
    model.eval()
    results={}

    for split in ["train","val"]:
        losses=[]

        for _ in range(20):
            inputs, targets=get_batch(split)
            predictions=model(inputs)

            batch_loss=F.cross_entropy(
                predictions.reshape(-1, vocab_size),
                targets.reshape(-1)
            )
            losses.append(batch_loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results

model.train()

max_steps = 10000
time_limit_seconds = 15 * 60

model.train()
training_start = time.perf_counter()
completed_steps = 0
best_val_loss = float("inf")
best_state = None
evaluations_without_improvement = 0

for step in range(max_steps + 1):
    if step % 500 == 0:
        results = estimate_loss()
        elapsed = time.perf_counter() - training_start

        print(
            f"Step {step}: "
            f"train={results['train']:.4f}, "
            f"val={results['val']:.4f}, "
            f"elapsed={elapsed:.1f}s",
            flush=True
        )

        if results["val"] < best_val_loss:
            best_val_loss = results["val"]
            best_state = copy.deepcopy(model.state_dict())
            evaluations_without_improvement = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            evaluations_without_improvement += 1

        if evaluations_without_improvement >= 4:
            print("Stopping: no validation improvement for 4 evaluations.")
            break    

    if step == max_steps:
        break

    if time.perf_counter() - training_start >= time_limit_seconds:
        print("Training time limit reached.", flush=True)
        break

    inputs, targets = get_batch("train")
    predictions = model(inputs)

    loss = F.cross_entropy(
        predictions.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    completed_steps += 1

elapsed = time.perf_counter() - training_start
print(f"Completed {completed_steps} updates in {elapsed:.1f} seconds")

if best_state is not None:
    model.load_state_dict(best_state)

print(f"Restored best validation loss: {best_val_loss:.4f}")

@torch.no_grad()
def generate(model, token_ids, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):
        context=token_ids[:, -model.block_size:]
        logits=model(context)

        next_logits=logits[:, -1, :]
        probabilities=torch.softmax(next_logits, dim=1)
        next_id=torch.multinomial(probabilities, num_samples=1)

        token_ids=torch.cat([token_ids, next_id], dim=1)

    return token_ids

prompt="The "
prompt_ids=torch.tensor(
    [[stoi[character] for character in prompt]],
    dtype=torch.long,
    device=next(model.parameters()).device
)

generated_ids=generate(model, prompt_ids, max_new_tokens=200)
generated_text="".join(
    chars[token_id] for token_id in generated_ids[0].tolist()
)

print("\nGenerated text:")
print(generated_text)

checkpoint={
    "model_state": model.state_dict(),
    "config": {
        "vocab_size": model.token_embedding.num_embeddings,
        "block_size": model.block_size,
        "embedding_size": model.token_embedding.embedding_dim,
        "num_heads": len(model.blocks[0].attention.heads),
        "num_layers": len(model.blocks)
    },
    "chars": chars,
}

torch.save(checkpoint, "nature_gpt.pt")
print("Saved nature_gpt.pt")

saved=torch.load(
    "nature_gpt.pt",
    map_location="cpu",
    weights_only=True
)

loaded_model=TinyGPT(**saved["config"])
loaded_model.load_state_dict(saved["model_state"])

model.eval()
loaded_model.eval()

with torch.no_grad():
    original_logits=model(prompt_ids).cpu()
    loaded_logits=loaded_model(prompt_ids.cpu())

print(
    "Reload matches:",
    torch.allclose(original_logits, loaded_logits)
)
print("Vocabulary matches:", saved['chars'] == chars)

