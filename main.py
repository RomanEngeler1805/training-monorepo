from data.dataloader import DataLoader
from data.dataset import Dataset
from models.transformer import Model, Tokenizer


def main():
    print("Hello World!")

    # initialise the data loader
    dataset = Dataset(data_path="roneneldan/TinyStories", text_column="text")
    dataloader = DataLoader(dataset=dataset["train"], batch_size=2, shuffle=False)

    # initialise the tokenizer
    tokenizer = Tokenizer(tokenizer_name="google/gemma-3-270m")
    # initialise the model
    model = Model(model_name="google/gemma-3-270m")
    model.train()
    # loop through the data loader
    for batch in dataloader:
        # forward pass
        tokenized = tokenizer.tokenize(batch)
        output = model.forward(
            input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
        )
        print(output.logits)
        # update the model
        break


if __name__ == "__main__":
    main()
