import torch

from data.dataloader import DataLoader
from data.dataset import Dataset
from models.transformer import ScratchModel, Tokenizer
from training.ce_loss import CrossEntropy
from training.optimizer import SGD
from training.sft_trainer import SFTTrainer
from utils.utils import logger


def main():
    # Training
    n_epochs = 100
    n_samples = 256
    batch_size = 2
    lr = 2e-2
    lr_decay = 1.0
    max_grad_norm = 10.0

    # Model
    max_length = 512
    n_layers = 8
    d_model = 512
    num_heads = 4
    d_hidden = 512
    dropout = 0.0
    dtype = torch.bfloat16

    # initialise the data loader
    logger.info("Loading data...")
    dataset = Dataset(data_path="roneneldan/TinyStories", split="train", text_column="text")
    dataset.data = dataset.data.select(range(n_samples))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Loaded {len(dataset)} samples")

    # initialise the tokenizer & model
    logger.info("Loading model and tokenizer...")
    tokenizer = Tokenizer(tokenizer_name="google/gemma-3-270m")
    # model = Model(model_name="google/gemma-3-270m")
    model = ScratchModel(
        n_layers=n_layers,
        n_vocab=tokenizer.tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_hidden=d_hidden,
        dropout=dropout,
        dtype=dtype,
    )
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    num_params_millions = num_params / 1_000_000
    logger.info(f"Model ready for training - Parameters: {num_params_millions:.2f}M")

    # optimizer
    loss_fn = CrossEntropy()
    optimizer = SGD(model_parameters=model.parameters(), lr=lr, lr_decay=lr_decay)
    logger.info(f"Starting training: {n_epochs} epochs, batch_size={batch_size}, lr={lr}")

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        max_length=max_length,
        max_grad_norm=max_grad_norm,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    trainer.train(n_epochs=n_epochs)


if __name__ == "__main__":
    main()
