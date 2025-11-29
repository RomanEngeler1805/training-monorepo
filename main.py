import matplotlib.pyplot as plt
import torch.optim as optim

from data.dataloader import DataLoader
from data.dataset import Dataset
from models.transformer import Model, Tokenizer
from training.loss import CrossEntropy


def main():
    print("Hello World!")

    n_epochs = 10
    batch_size = 2
    lr = 1e-3
    losses = []

    # initialise the data loader
    dataset = Dataset(data_path="roneneldan/TinyStories", split="train", text_column="text")
    dataset.data = dataset.data.select(range(8))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # initialise the tokenizer & model
    tokenizer = Tokenizer(tokenizer_name="google/gemma-3-270m")
    model = Model(model_name="google/gemma-3-270m")
    model.train()

    # training
    loss_fn = CrossEntropy()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(n_epochs):
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
            print("loss", loss_value)
            losses.append(loss_value)

            # Add these three lines for optimizer
            optimizer.zero_grad()  # Zero out gradients from previous iteration
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

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
