from datasets import load_dataset  # type: ignore


class Dataset:
    def __init__(self, data_path: str, text_column: str):
        self.data = load_dataset(data_path)
        self.text_column = text_column

    def __len__(self):
        """return the number of items in the dataset"""
        return len(self.data)

    def __getitem__(self, index: int):
        """return the item at the given index"""
        return self.data[index][self.text_column]
