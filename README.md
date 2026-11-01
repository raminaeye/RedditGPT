# RedditGPT

A small GPT-style transformer trained from scratch in PyTorch on r/AskScience posts, in the style of nanoGPT, with a byte-level BPE tokenizer trained on the same text.

## What's inside

- `notebooks/redditGPT.ipynb` prepares data, trains and samples
- `utils/` holds the attention blocks and helpers

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/redditGPT.ipynb
```

The notebook needs your own `data/askscience_data.csv`; it is not included. On a fresh clone, comment out the `utils.helper` and `utils.attention_block` imports in the first cell, run the data cells that write the files in `data/`, then run the imports.

## Related tutorial

- [Transformers](https://raminaeye.github.io/transformers.html)
