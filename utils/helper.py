"""Training helpers: batch sampling and periodic loss estimation.

Note: importing this module loads the pickled config, training data, and
validation data from the data directory (see ``path_to_data``), so those
artifacts must exist before import.
"""

import pandas as pd
from collections import Counter
import sys
import numpy as np
from matplotlib.pylab import plt
pd.options.mode.chained_assignment = None
from IPython.display import Image, display
import torch
import torch.nn as nn
from torch.nn import functional as F
import ast
from dataclasses import dataclass
from contextlib import nullcontext
import pickle, dill
from unidecode import unidecode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_to_data = '../data/'
eval_iters = 50

@dataclass
class GPTConfig:
    block_size: int = 32
    vocab_size: int = 1800
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 512
    dropout: float = 0.2
    batch_size: int = 32
    temperature=1.0
    top_k=20

config = dill.load(open(path_to_data + 'scratch_reddit_gpt_config.pickle','rb'))

train_data = pickle.load(open(path_to_data + 'train_data.pickle','rb'))
val_data = pickle.load(open(path_to_data + 'valid_data.pickle','rb'))

# Training Helper
@torch.no_grad()
def estimate_loss(model):
    """Estimate average train and validation loss over ``eval_iters`` batches.

    Switches the model to eval mode, samples random batches from each split,
    then restores train mode. Used to track progress during training.

    Args:
        model: A ``GPTLanguageModel`` instance whose ``forward`` returns
            ``(logits, loss)``.

    Returns:
        Dict with keys 'train' and 'val' holding the mean loss tensors.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_batch(split):
    """Sample a random batch of (input, target) token sequences.

    Targets are the inputs shifted one token to the right, so the model
    learns to predict the next token at every position.

    Args:
        split: 'train' or 'val'.

    Returns:
        Tuple ``(x, y)`` of long tensors with shape
        ``(batch_size, block_size)`` on the active device.
    """
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (config.batch_size,)) # randomly pick a data sample
    x = torch.stack([data[i][0:-1] for i in ix])
    y = torch.stack([data[i][1:] for i in ix]) # label is shifted by 1
    x, y = x.to(device), y.to(device)
    return x, y
