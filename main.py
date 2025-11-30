import logging

import matplotlib.pyplot as plt

from data.dataloader import DataLoader
from data.dataset import Dataset
from models.transformer import Model, Tokenizer
from training.loss import CrossEntropy
from training.optimizer import SGD

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    n_epochs = 10
    batch_size = 2
    lr = 1e-3
    losses = []

    # initialise the data loader
    logger.info("Loading data...")
    dataset = Dataset(data_path="roneneldan/TinyStories", split="train", text_column="text")
    dataset.data = dataset.data.select(range(8))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Loaded {len(dataset)} samples")

    # initialise the tokenizer & model
    logger.info("Loading model and tokenizer...")
    tokenizer = Tokenizer(tokenizer_name="google/gemma-3-270m")
    model = Model(model_name="google/gemma-3-270m")
    model.train()
    logger.info("Model ready for training")

    # optimizer
    loss_fn = CrossEntropy()
    optimizer = SGD(model_parameters=model.parameters(), lr=lr)
    logger.info(f"Starting training: {n_epochs} epochs, batch_size={batch_size}, lr={lr}")

    for epoch in range(n_epochs):
        epoch_losses = []

        # loop through the data loader
        for batch in dataloader:
            # forward pass
            tokenized = tokenizer.tokenize(batch).to(device=model.device)
            output = model.forward(
                input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
            )
            # update the model
            loss = loss_fn.calculate_loss(
                preds=output.logits[:, :-1, :], labels=tokenized["input_ids"][:, 1:]
            )
            loss_value = loss.detach().cpu().item()

            logger.debug(f"Epoch {epoch}, Loss: {loss_value:.4f}")

            epoch_losses.append(loss_value)
            losses.append(loss_value)

            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
