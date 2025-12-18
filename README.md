# Training Monorepo

A minimal transformer training framework with implementations of supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO). Built from scratch with PyTorch.

## Features

- **Transformer models**: Custom implementation or HuggingFace models
- **Training regimes**: SFT and GRPO with reward-based optimization
- **Decoding**: Greedy and beam search strategies
- **Data loading**: Local JSON/JSONL files or HuggingFace datasets

## Quick Start

```bash
# Install dependencies
uv sync

# Run training
uv run python main.py
```

Checkout the Makefile for useful commands.

Configure training in `main.py` by selecting:
- `training_regime`: `"sft"` or `"grpo"`
- `model`: `"SimpleTransformer"` (from scratch) or a HuggingFace model name
- Dataset path (local file or HuggingFace Hub)

## Project Structure

```
├── models/          # Transformer architecture (attention, embeddings, feedforward)
├── training/        # SFT and GRPO trainers, loss functions, optimizers
├── data/            # Dataset and dataloader implementations
├── inference/       # Decoding strategies (greedy, beam search)
└── tests/           # Unit tests
```

## To-Do

- [ ] Data analysis
- [ ] Dockerize
- [ ] Set up terraform
- [ ] MLFlow monitoring