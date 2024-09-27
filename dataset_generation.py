import random
import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate():
    sequences = []
    # Постоянное значение. Например: 1 1 1, 3 3 3 3
    for i in range(10):
        for j in range(3, 7):
            sequences.append([i] * j)

    # Простое возрастание. Например: 0 1 2 3, 5 6 7 8
    for i in range(7):
        for j in range(3, 10 - i):
            sequences.append(list(range(i, i + j + 1)))

    # Простое убывание. Например: 4 3 2, 9 8 7 6 5
    for i in range(7):
        for j in range(3, 10 - i):
            sequences.append(list(range(i + j, i - 1, -1)))

    # Чередование 2 цифр. Например: 1 2 1 2, 6 8 6
    for i in range(10):
        for j in range(10):
            for k in range(2, 5):
                sequences.append([i, j] * k)

    # Чередование 3 цифр. Например: 1 4 5 1 4, 7 5 1 7 5 1 7
    sequences = sequences * 2
    for i in range(10):
        for j in range(10):
            for f in range(10):
                sequences.append(([i, j, f] * 3)[:-random.randint(1, 4)])

    input_sequences = []
    target_sequences = []
    for seq in sequences:
        input_sequences.append(seq[:-1])
        target_sequences.append(seq[-1])

    max_seq_len = max(len(seq) for seq in input_sequences)
    n_features = 10

    input_sequences_padded = np.zeros((len(input_sequences), max_seq_len, n_features), dtype=np.float32)
    target_sequences_one_hot = np.zeros((len(target_sequences), n_features), dtype=np.float32)

    for i, seq in enumerate(input_sequences):
        for j, num in enumerate(seq):
            input_sequences_padded[i, j, num] = 1
        target_sequences_one_hot[i, target_sequences[i]] = 1

    input_sequences_padded = torch.tensor(input_sequences_padded)
    target_sequences_one_hot = torch.tensor(target_sequences_one_hot)

    return TensorDataset(input_sequences_padded, target_sequences_one_hot)
