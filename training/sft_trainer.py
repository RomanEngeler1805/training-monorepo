import matplotlib.pyplot as plt
import torch

from data.dataloader import DataLoader
from models.transformer import Model, ScratchModel, Tokenizer
from training.ce_loss import CrossEntropy
from training.optimizer import SGD
from utils.utils import logger


class SFTTrainer:
    def __init__(
        self,
        model: Model | ScratchModel,
        tokenizer: Tokenizer,
        dataloader: DataLoader,
        max_length: int,
        optimizer: torch.optim.Optimizer | SGD,
        loss_fn: CrossEntropy,
        max_grad_norm: float = 10.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.max_length = max_length
        self.max_grad_norm = max_grad_norm

    def train(
        self,
        n_epochs: int = 3,
        plot_losses: bool = True,
        log_interval: int = 10,
    ) -> list[float]:
        """Train the model for specified number of epochs.

        Args:
            n_epochs: Number of training epochs
            plot_losses: Whether to plot training losses at the end
            log_interval: Log loss every N batches

        Returns:
            List of training losses for each batch
        """
        self.model.train()
        losses = []

        for epoch in range(n_epochs):
            epoch_losses = []

            # loop through the data loader
            for batch_idx, batch in enumerate(self.dataloader):
                # Forward pass
                prompts = [item["prompt"] for item in batch]
                tokenized = self.tokenizer.tokenize(input=prompts, max_length=self.max_length).to(
                    device=self.model.device
                )

                output = self.model.forward(
                    input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
                )

                # Calculate loss
                loss_mask = tokenized["attention_mask"][:, 1:]
                loss = self.loss_fn.calculate_loss(
                    preds=output.logits[:, :-1, :],
                    labels=tokenized["input_ids"][:, 1:],
                    attention_mask=loss_mask,
                )
                loss_value = loss.detach().cpu().item()

                if batch_idx % log_interval == 0:
                    logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_value:.4f}")

                epoch_losses.append(loss_value)
                losses.append(loss_value)

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                # Cleanup
                del output, loss, tokenized
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            logger.info(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}")

        logger.info("Training completed")

        if plot_losses:
            self._plot_losses(losses)

        return losses

    def _plot_losses(self, losses: list[float]):
        """Plot training losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.show()
