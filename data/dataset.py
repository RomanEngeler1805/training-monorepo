from datasets import load_dataset  # type: ignore[import-untyped]


class Dataset:
    def __init__(
        self, data_path: str, split: str, text_column: str, solution_column: str | None = None
    ):
        self.data = load_dataset(data_path)[split]
        self.text_column = text_column
        self.solution_column = solution_column

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
