with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))
print(text[:500])