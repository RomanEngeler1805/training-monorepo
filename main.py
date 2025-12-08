import logging

import matplotlib.pyplot as plt
import torch

from data.dataloader import DataLoader
from data.dataset import Dataset
from models.transformer import ScratchModel, Tokenizer
from training.loss import CrossEntropy
from training.optimizer import SGD

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    # Training
    n_epochs = 1000
    n_samples = 256
    batch_size = 4
    lr = 2e-2
    lr_decay = 1.0
    losses = []

    # Model
    max_length = 512
    n_layers = 12
    d_model = 512
    num_heads = 4
    d_hidden = 512
    dropout = 0.0
    dtype = torch.float32

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

    for epoch in range(n_epochs):
        epoch_losses = []

        # loop through the data loader
        for batch in dataloader:
            # forward pass
            tokenized = tokenizer.tokenize(input=batch, max_length=max_length).to(
                device=model.device
            )
            output = model.forward(
                input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
            )
            # update the model
            loss_mask = tokenized["attention_mask"][:, 1:]
            loss = loss_fn.calculate_loss(
                preds=output.logits[:, :-1, :],
                labels=tokenized["input_ids"][:, 1:],
                attention_mask=loss_mask,
            )
            loss_value = loss.detach().cpu().item()

            logger.debug(f"Epoch {epoch}, Loss: {loss_value:.4f}")

            epoch_losses.append(loss_value)
            losses.append(loss_value)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # Explicit cleanup of intermediate tensors
            del output, loss, tokenized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Epoch-level summary at INFO level
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}")

    logger.info("Training completed")

    # Plot the training losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
