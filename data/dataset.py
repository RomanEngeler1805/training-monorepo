import json
import os

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore


class Dataset:
    def __init__(
        self,
        data_path: str,
        text_column: str,
        split: str | None = None,
        solution_column: str | None = None,
    ):
        self.text_column = text_column
        self.solution_column = solution_column
        self.data = self.load_dataset(data_path=data_path, split=split)

    def load_dataset(self, data_path: str, split: str | None = None) -> HFDataset:
        path = os.path.join(os.getcwd(), data_path)
        if os.path.exists(path):
            if path.endswith(".jsonl"):
                with open(path) as f:
                    # JSONL format: one JSON object per line
                    data = []
                    for line in f:
                        if line.strip():  # Skip empty lines
                            data.append(json.loads(line))
            elif path.endswith(".json"):
                with open(path) as f:
                    # JSON format: single JSON array or object
                    data = json.load(f)
            else:
                raise ValueError("only json and jsonl loading from local dir implemented")

            # If data is a single dict, wrap it in a list
            if isinstance(data, dict):
                data = [data]

            dataset = HFDataset.from_list(data)
            return dataset

        # Try loading from HuggingFace Hub
        try:
            if split:
                dataset = load_dataset(data_path, split=split)
            else:
                dataset_dict = load_dataset(data_path)
                split_name = list(dataset_dict.keys())[0]
                dataset = dataset_dict[split_name]
            return dataset
        except Exception as e:
            raise ValueError(f"Could not load dataset from path '{data_path}': {e}") from e

    def __len__(self):
        """return the number of items in the dataset"""
        return len(self.data)

    def __getitem__(self, index: int):
        """return the item at the given index"""
        item = self.data[index][self.text_column]
        # Extract content if item is a list of dicts (e.g., [{"content": "..."}])
        if (
            isinstance(item, list)
            and len(item) > 0
            and isinstance(item[0], dict)
            and "content" in item[0]
        ):
            prompt = item[0]["content"]
        else:
            prompt = item

        # If solution_column is provided, return dict with both prompt and solution
        if self.solution_column is not None:
            solution = self.data[index][self.solution_column]
            return {"prompt": prompt, "solution": solution}

        # Otherwise, return just the text (for next token prediction)
        return {"prompt": prompt}
