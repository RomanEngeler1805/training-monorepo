import torch

from training.ce_loss import CrossEntropy


class TestCrossEntropy:
    def test_cross_entropy_init(self):
        """Test that CrossEntropy initializes correctly"""
        loss_fn = CrossEntropy()
        assert loss_fn.loss is not None

    def test_calculate_loss_correct_prediction(self):
        """Test basic loss calculation"""
        loss_fn = CrossEntropy()
        labels = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.long)
        preds = torch.tensor(
            [
                [
                    [100.0, 0.0, 0.0],
                    [0.0, 100.0, 0.0],
                    [100.0, 0.0, 0.0],
                ]
            ]
        )
        loss = loss_fn.calculate_loss(preds, labels)
        assert loss < 0.001

    def test_calculate_loss_wrong_prediction(self):
        """Test basic loss calculation"""
        loss_fn = CrossEntropy()
        labels = torch.tensor(
            [
                [
                    1.0,
                    0.0,
                ]
            ],
            dtype=torch.long,
        )
        preds = torch.tensor(
            [
                [
                    [100.0, 0.0, 0.0],
                    [0.0, 100.0, 0.0],
                ]
            ]
        )
        loss = loss_fn.calculate_loss(preds, labels)
        assert loss > 10.0

    def test_calculate_loss_batch(self):
        """Test loss calculation with batch x sequence_length x vocab_size"""
        loss_fn = CrossEntropy()

        # Predictions: (batch_size=2, sequence_length=3, vocab_size=5)
        preds = torch.tensor(
            [
                [
                    [10.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 10.0, 0.0, 0.0],
                ],
                # Batch 1: 3 tokens
                [
                    [0.0, 0.0, 0.0, 10.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 10.0],
                    [10.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )

        # Labels: (batch_size=2, sequence_length=3)
        labels = torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 0],
            ],
            dtype=torch.long,
        )

        loss = loss_fn.calculate_loss(preds, labels)

        # With high logits for correct classes, loss should be very low
        assert loss < 0.01
        assert loss.item() > 0  # Loss should be positive
        assert isinstance(loss, torch.Tensor)
