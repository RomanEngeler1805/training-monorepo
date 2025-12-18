from __future__ import annotations

import copy

import torch

from data.dataloader import DataLoader
from data.dataset import Dataset
from inference.decoding import BeamDecoder
from models.transformer import Model, ScratchModel, Tokenizer
from training.accuracy_rewards import accuracy_reward
from training.ce_loss import CrossEntropy
from training.grpo_trainer import GRPOTrainer
from training.optimizer import SGD
from training.sft_trainer import SFTTrainer
from utils.utils import logger


def main(config: dict):
    # Dataloader =========
    logger.info("Loading data...")
    dataset = Dataset(
        data_path=config["dataset"],
        text_column=config["text_column"],
        split=config.get("split"),
        solution_column=config.get("solution_column"),
    )
    dataset.data = dataset.data.select(range(min(config["num_samples"], len(dataset))))
    dataloader = DataLoader(dataset=dataset, batch_size=config["batch_size"], shuffle=True)
    logger.info(f"Loaded {len(dataset)} samples")

    # Tokenizer & Model =========
    logger.info("Loading model and tokenizer...")

    model: Model | ScratchModel
    if config["model"] == "SimpleTransformer":
        tokenizer = Tokenizer(tokenizer_name="google/gemma-3-270m-it")

        model = ScratchModel(
            n_layers=config["n_layers"],
            n_vocab=tokenizer.tokenizer.vocab_size,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_hidden=config["d_hidden"],
            dropout=config["dropout"],
            dtype=config["dtype"],
        )
    else:
        tokenizer = Tokenizer(tokenizer_name=config["model"])
        model = Model(model_name=config["model"], dtype=config["dtype"])

    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    num_params_millions = num_params / 1_000_000
    logger.info(f"Model ready for training - Parameters: {num_params_millions:.2f}M")

    # Optimizer =========
    optimizer = SGD(
        model_parameters=model.parameters(), lr=config["lr"], lr_decay=config["lr_decay"]
    )
    logger.info(
        f"Starting training: {config['n_epochs']} epochs, batch_size={config['batch_size']}, lr={config['lr']}"
    )

    # Trainer =========
    trainer: SFTTrainer | GRPOTrainer
    if config["training_regime"] == "sft":
        loss_fn = CrossEntropy()
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            max_length=config["max_length"],
            max_grad_norm=config["max_grad_norm"],
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
    elif config["training_regime"] == "grpo":
        if config["beta"] > 0.0:
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
        else:
            ref_model = None
        decoder = BeamDecoder(num_beams=config["num_beams"])

        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            max_length=config["max_length"],
            max_new_tokens=config["max_new_tokens"],
            eps=config["eps"],
            beta=config["beta"],
            optimizer=optimizer,
            decoder=decoder,
            reward_fn=accuracy_reward,
            max_grad_norm=config["max_grad_norm"],
        )
    else:
        raise ValueError("Training regime not recognised.")

    trainer.train(n_epochs=config["n_epochs"])


if __name__ == "__main__":
    sft_config = {
        "training_regime": "sft",
        "dataset": "roneneldan/TinyStories",
        "split": "train",
        "text_column": "text",
        "num_samples": 64,
        "n_epochs": 100,
        "batch_size": 8,
        "lr": 2e-2,
        "lr_decay": 1.001,
        "max_grad_norm": 10.0,
        "max_length": 512,
        "model": "SimpleTransformer",
        "n_layers": 12,
        "d_model": 512,
        "num_heads": 4,
        "d_hidden": 512,
        "dropout": 0.0,
        "dtype": torch.bfloat16,
    }
    grpo_config = {
        "training_regime": "grpo",
        "dataset": "raw/toy_rpo_dataset.jsonl",  # "trl-lib/DeepMath-103K" -> "prompt", "solution"
        "text_column": "prompt",
        "solution_column": "solution",
        "num_samples": 16,
        "n_epochs": 5,
        "batch_size": 2,
        "lr": 1e-4,
        "lr_decay": 1.00,
        "max_grad_norm": 10.0,
        "eps": 0.1,
        "beta": 0.0,
        "num_beams": 8,
        "max_new_tokens": 256,  # Maximum number of new tokens to generate
        "max_length": 512,  # Maximum total sequence length (tokenizer)
        "model": "google/gemma-3-270m-it",
        "n_layers": 12,
        "d_model": 512,
        "num_heads": 4,
        "d_hidden": 512,
        "dropout": 0.0,
        "dtype": torch.bfloat16,
    }

    main(config=sft_config)
