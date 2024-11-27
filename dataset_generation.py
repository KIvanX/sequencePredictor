import random
import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate():
    sequences = []
    # Постоянное значение. Например: 1 1 1, 3 3 3 3
    for i in range(30):
        for j in range(3, 30):
            sequences.append([i] * j)

    # Арифметическое возрастание. Например: 0 1 2 3, 5 7 9 11
    for d in range(1, 4):
        for i in range(25):
            for j in range(3, 30 - i):
                if len(range(i, i + j + 1, d)) > 3:
                    sequences.append(list(range(i, i + j + 1, d)))

    # Арифметическое убывание. Например: 4 3 2, 9 8 7 6 5, 10, 8, 6, 4
    for d in range(1, 4):
        for i in range(30):
            for j in range(3, 30 - i):
                if len(range(i + j, i - 1, -d)) > 3:
                    sequences.append(list(range(i + j, i - 1, -d)))

    # Чередование 2 цифр. Например: 1 2 1 2, 6 8 6
    for i in range(20):
        for j in range(20):
            for k in range(2, 13):
                sequences.append([i, j] * k)

    # Чередование 3 цифр. Например: 1 4 5 1 4, 7 5 1 7 5 1 7
    sequences = sequences * 2
    for i in range(20):
        for j in range(20):
            for f in range(20):
                sequences.append(([i, j, f] * random.randint(3, 6))[:-random.randint(1, 4)])

    input_sequences = []
    target_sequences = []
    for seq in sequences:
        input_sequences.append(seq[:-1])
        target_sequences.append(seq[-1])

    max_seq_len = max(len(seq) for seq in input_sequences)
    n_features = 30

    input_sequences_padded = np.zeros((len(input_sequences), max_seq_len, n_features), dtype=np.float32)
    target_sequences_one_hot = np.zeros((len(target_sequences), n_features), dtype=np.float32)

    for i, seq in enumerate(input_sequences):
        for j, num in enumerate(seq):
            input_sequences_padded[i, j, num] = 1
        target_sequences_one_hot[i, target_sequences[i]] = 1

    input_sequences_padded = torch.tensor(input_sequences_padded)
    target_sequences_one_hot = torch.tensor(target_sequences_one_hot)

    return TensorDataset(input_sequences_padded, target_sequences_one_hot)