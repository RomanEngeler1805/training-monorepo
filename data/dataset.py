from datasets import load_dataset  # type: ignore


class Dataset:
    def __init__(self, data_path: str):
        self.data = load_dataset(data_path)

    def __len__(self):
        """return the number of items in the dataset"""
        return len(self.data)

    def __getitem__(self, index: int):
        """return the item at the given index"""
        return self.data[index]
