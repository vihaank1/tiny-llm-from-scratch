import torch

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))
print(text[:500])


chars = sorted(set(text))
vocab_size = len(chars)

print("Vocabulary:", chars)
print("Vocabulary size:", vocab_size)

stoi = {}

for index, character in enumerate(chars):
    stoi[character] = index

print(stoi)

encoded =[]

for character in text:
    encoded.append(stoi[character])

print("First 10 IDs:", encoded[:10])
print("Encoded length:", len(encoded))

data = torch.tensor(encoded, dtype=torch.long)

split_index=int(0.9*len(data))

train_data=data[:split_index]
val_data=data[split_index:]

print("Training shape:", train_data.shape)
print("Validation shape:", val_data.shape)

block_size = 8
batch_size = 4


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