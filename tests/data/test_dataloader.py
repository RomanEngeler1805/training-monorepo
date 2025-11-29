from unittest.mock import MagicMock

import pytest

from data.dataloader import DataLoader


@pytest.fixture
def mock_dataset():
    """Fixture that creates a mock dataset with 10 items"""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    # Make __getitem__ return items with index for easier testing
    dataset.__getitem__ = MagicMock(side_effect=lambda idx: {"text": f"item_{idx}", "index": idx})
    return dataset


class TestDataLoader:
    def test_dataloader_init(self, mock_dataset):
        """Test DataLoader initialization with custom parameters"""
        loader = DataLoader(mock_dataset, batch_size=3, shuffle=False)
        assert loader.batch_size == 3
        assert loader.shuffle is False
        assert len(loader.indices) == 10

    def test_dataloader_len_batch_size_smaller_than_dataset(self, mock_dataset):
        """Test __len__ when batch_size is larger than dataset"""
        loader = DataLoader(mock_dataset, batch_size=2)
        assert len(loader) == 5

    def test_dataloader_len_batch_size_larger_than_dataset(self, mock_dataset):
        """Test __len__ when batch_size is larger than dataset"""
        loader = DataLoader(mock_dataset, batch_size=12)
        assert len(loader) == 1

    def test_dataloader_iteration_batch_size_3(self, mock_dataset):
        """Test iteration with batch_size=3"""
        loader = DataLoader(mock_dataset, batch_size=3, shuffle=False)
        batches = list(loader)
        assert len(batches) == 4  # 10 items / 3 = 4 batches
        # First batch should have 3 items
        assert len(batches[0]) == 3
        assert batches[0][0]["index"] == 0
        assert batches[0][1]["index"] == 1
        assert batches[0][2]["index"] == 2
        # Last batch should have 1 item (10 % 3 = 1)
        assert len(batches[3]) == 1
        assert batches[3][0]["index"] == 9

    def test_dataloader_multiple_iterations(self, mock_dataset):
        """Test that DataLoader can be iterated multiple times"""
        loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)
        batches1 = list(loader)
        batches2 = list(loader)
        # Both iterations should return the same number of batches
        assert len(batches1) == len(batches2)
        # Both should have the same items (with shuffle=False)
        indices1 = [item["index"] for batch in batches1 for item in batch]
        indices2 = [item["index"] for batch in batches2 for item in batch]
        assert indices1 == indices2

    def test_dataloader_reset(self, mock_dataset):
        """Test __reset__ method resets the current index"""
        loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)
        # Iterate through some batches
        _ = next(iter(loader))
        _ = next(iter(loader))
        # Reset and iterate again
        loader.__reset__()
        iter_loader = iter(loader)
        batch_after_reset = next(iter_loader)
        # Should start from the beginning again
        assert batch_after_reset[0]["index"] == 0
