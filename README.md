# RedditGPT

Train a small GPT-style transformer from scratch on Reddit text, following Andrej Karpathy's nanoGPT approach. The model is trained on post titles and bodies from r/AskScience to predict the next token in a sequence.

No external APIs are used anywhere in this repo. There are no Reddit API calls and no OpenAI calls, so no API keys or credentials are needed. You only need the training data files described below.

## Repository structure

- `notebooks/redditGPT.ipynb` - the main notebook: data cleaning, tokenizer training, sequence preparation, model training, and text generation.
- `utils/attention_block.py` - the transformer model (`GPTLanguageModel`) with multi-head self-attention blocks, plus tokenizer setup.
- `utils/helper.py` - training helpers: random batch sampling (`get_batch`) and periodic train/val loss estimation (`estimate_loss`).
- `utils/data_utility.py` - text cleaning utilities for preparing the Reddit corpus.
- `snapshots/` - figures used in the notebook (attention diagrams).
- `requirements.txt` - Python dependencies.

Training artifacts (tokenizer files, pickled config, encoded train/val tensors, model checkpoints) live in a `data/` directory at the repo root. That directory is gitignored because of its size; the notebook creates what it needs from the raw CSV.

## How to run it

1. Create a Python environment (3.8+) and install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Place your data in `data/`:
   - `askscience_data.csv` with `title` and `body` columns (or skip the cleaning cells and provide your own `training_text.txt`, one sentence per line wrapped in `<s>` ... `</s>`).

3. Open `notebooks/redditGPT.ipynb` and run the cells top to bottom:
   - **Data**: load and clean the Reddit corpus, write `training_text.txt`.
   - **Tokenizer**: train a byte-level BPE tokenizer on the corpus.
   - **Hyperparameters**: set model size and training settings in `GPTConfig`.
   - **Data prep**: chunk, encode, pad, and split into train/validation sets.
   - **Self Attention**: a small worked example of scaled dot-product attention with a causal mask.
   - **Model training**: train the transformer with AdamW, printing loss and samples along the way.
   - **Decoding**: generate new text from a prompt with temperature and top-k sampling.

Set `FromDrive = 1` in the data cell to load a previously saved checkpoint instead of training from scratch. Training runs on CUDA when available and falls back to CPU.

## What you learn

- How a decoder-only transformer is built from scratch: token embeddings, causal multi-head self-attention, feed-forward blocks, and residual connections.
- How to train a byte-level BPE tokenizer on your own corpus and prepare padded, fixed-length training sequences.
- How next-token training works in practice: shifted targets, masked padding in the loss, batch sampling, and train/val loss tracking.
- How sampling works at generation time: temperature scaling, top-k filtering, and early stopping on the end-of-sequence token.
