import numpy as np


data = np.memmap(
    "datasets/train.bin",
    dtype=np.uint32,
    mode="r"
)


print("Tokens:", len(data))

print("First 100 tokens:")
print(data[:100])
