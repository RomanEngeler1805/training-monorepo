from data.dataloader import DataLoader
from data.dataset import Dataset
from models.transformer import Model


def main():
    print("Hello World!")

    # initialise the data loader
    dataset = Dataset(data_path="roneneldan/TinyStories")
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    print(len(dataloader))
    # initialise the tokenizer
    # tokenizer = Tokenizer(tokenizer_name="google/gemma-3-270m")
    # initialise the model
    model = Model(model_name="google/gemma-3-270m")
    model.train()
    # loop through the data loader
    # forward pass
    # backward pass
    # update the model


if __name__ == "__main__":
    main()
