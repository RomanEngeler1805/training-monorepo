from unittest.mock import MagicMock, patch

import pytest

from data.dataset import Dataset


@pytest.fixture
def mock_dataset_data():
    """Fixture that creates a mock dataset object"""
    mock_data = MagicMock()
    mock_data.__len__ = MagicMock(return_value=10)
    mock_data.__getitem__ = MagicMock(return_value={"text": "sample"})
    return mock_data


@pytest.fixture
def dataset(mock_dataset_data):
    """Fixture that creates a Dataset instance with mocked load_dataset"""
    with patch("data.dataset.load_dataset", return_value=mock_dataset_data):
        yield Dataset("fake/path")


class TestDataset:
    def test_dataset_init(self, mock_dataset_data):
        """Test that Dataset initializes with mocked load_dataset"""
        with patch("data.dataset.load_dataset", return_value=mock_dataset_data):
            dataset = Dataset("fake/path")
            assert dataset.data == mock_dataset_data

    def test_dataset_len(self, dataset):
        """Test that Dataset.__len__ returns the correct length"""
        assert len(dataset) == 10

    def test_dataset_getitem(self, dataset):
        """Test that Dataset.__getitem__ returns the respective item"""
        assert dataset[0] == {"text": "sample"}
