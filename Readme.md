# LLM Data Pipeline — Lab Assignment 5

## Overview
This lab builds a complete **data pipeline for training a Large Language Model (LLM)**.  
It takes raw text data and transforms it into batched, tokenized, fixed-length sequences that are ready to be fed into a language model for training.

No model training happens here — this is purely the **data preparation stage**.

---

## What This Lab Does

```
Raw Text → Tokenize → Chunk into Blocks → Batch → Ready for LLM Training
```

| Step | What Happens |
|------|-------------|
| 1. Load Dataset | Download AG News dataset from HuggingFace |
| 2. Train/Val Split | Split data 90% train / 10% validation |
| 3. Tokenize | Convert text to token IDs using DistilGPT2 tokenizer |
| 4. Group into Blocks | Concatenate and slice into 256-token chunks |
| 5. Create DataLoaders | Batch and shuffle data for training |
| 6. Visualize | Plot token length distribution of the dataset |

---

## Modifications from Original Lab

| | Original | This Version |
|---|---|---|
| Dataset | WikiText-2 | **AG News** |
| Tokenizer | GPT-2 | **DistilGPT2** |
| Block Size | 128 tokens | **256 tokens** |
| Validation Set | ❌ None | **✅ 90/10 split** |
| Data Visualization | ❌ None | **✅ Token length histogram** |

### Why These Changes?
- **AG News** — Real-world news dataset, more varied vocabulary than Wikipedia
- **DistilGPT2** — Same vocabulary as GPT-2 but 40% smaller and faster to experiment with
- **Block size 256** — News text has longer context, 256 tokens captures it better
- **Validation split** — Essential in any real MLOps pipeline to monitor overfitting
- **Visualization** — Best practice to understand your data before training

---

## Dataset

**AG News** — A news topic classification dataset containing 120,000 training examples across 4 categories: World, Sports, Business, and Science/Technology.

- Source: [HuggingFace Datasets](https://huggingface.co/datasets/ag_news)
- Train size: ~108,000 examples (after 90/10 split)
- Validation size: ~12,000 examples

---

## Model / Tokenizer

**DistilGPT2** — A distilled version of GPT-2 by HuggingFace.
- Same tokenizer vocabulary as GPT-2 (50,257 tokens)
- Uses Byte-Pair Encoding (BPE)
- Pad token set to EOS token (GPT-2 has no pad token by default)

---

## Requirements

```
torch
transformers
datasets
matplotlib
```

Install with:
```bash
pip install transformers datasets torch matplotlib
```

---

## How to Run

### Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Run the install cell first:
   ```python
   !pip install transformers datasets torch matplotlib
   ```
4. Copy and run each cell in order

### Local (Jupyter Notebook)
```bash
pip install transformers datasets torch matplotlib jupyter
jupyter notebook Lab1.ipynb
```

---

## Output

After running all cells you should see:

```
Total examples in dataset: 120000
Train size: 108000, Validation size: 12000
Vocabulary size: 50257
LM training sequences:   XXXXX
LM validation sequences: XXXXX

=== Train Loader Check ===
input_ids shape: torch.Size([8, 256])
labels shape:    torch.Size([8, 256])

=== Validation Loader Check ===
input_ids shape: torch.Size([8, 256])
labels shape:    torch.Size([8, 256])
```

A token length distribution plot is also saved as `token_length_distribution.png`.

---

## Key Concepts

**Tokenization** — Converting raw text into integer token IDs that a model can process.

**Block Size** — Fixed sequence length that the model expects as input. All sequences must be the same size.

**Causal Language Modeling** — The task of predicting the next token. Labels = Input IDs (the model's loss function handles the internal shift).

**DataLoader** — PyTorch utility that handles batching, shuffling, and efficient data delivery during training.

---

## File Structure

```
├── Lab1.ipynb                      # Main notebook
├── README.md                       # This file
└── token_length_distribution.png   # Generated plot
```

---

## Author
Lab Assignment 5 — MLOps Course  
Modified from original lab by [raminmohammadi](https://github.com/raminmohammadi/MLOps)
