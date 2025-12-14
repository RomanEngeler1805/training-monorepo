import math
from unittest.mock import MagicMock

import pytest
import regex as re
import torch

from inference.decoding import BeamDecoder
from models.transformer import ScratchModel, Tokenizer
from training.grpo_trainer import GRPOTrainer
from training.optimizer import SGD


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    return ScratchModel(
        n_layers=2,
        n_vocab=1000,
        d_model=64,
        num_heads=4,
        d_hidden=128,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def tokenizer():
    return Tokenizer(tokenizer_name="google/gemma-3-270m-it")


@pytest.fixture
def decoder():
    return BeamDecoder(num_beams=3)


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader (not actually used in _generate_completions test)."""
    return MagicMock()


@pytest.fixture
def trainer(small_model, tokenizer, decoder, mock_dataloader):
    """Create a GRPOTrainer instance for testing."""
    # Create a reference model (copy of the model)
    import copy

    ref_model = copy.deepcopy(small_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = SGD(model_parameters=small_model.parameters(), lr=1e-3)

    # Create a simple mock reward function
    def mock_reward_fn(completions):
        # Return a reward of 1.0 for each completion
        return [1.0] * len(completions)

    return GRPOTrainer(
        model=small_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataloader=mock_dataloader,
        max_length=20,
        max_new_tokens=20,
        eps=0.1,
        beta=0.01,
        optimizer=optimizer,
        decoder=decoder,
        reward_fn=mock_reward_fn,
    )


class TestGRPOTrainer:
    def test_generate_completions(self, trainer):
        """Test _generate_completions() tokenization and decoding functionality."""
        prompts = ["Hello world", "How are you"]

        tokenized_prompt, tokenized_prompt_completions, prompt_completions = (
            trainer._generate_completions(prompts)
        )

        # Verify tokenized_prompt structure
        assert "input_ids" in tokenized_prompt
        assert "attention_mask" in tokenized_prompt
        assert isinstance(tokenized_prompt["input_ids"], torch.Tensor)

        # Verify tokenized_prompt shapes (batch_size, prompt_length)
        batch_size = len(prompts)
        assert tokenized_prompt["input_ids"].shape[0] == batch_size

        # Verify generations shape (batch_size * num_beams, seq_length)
        num_beams = trainer.decoder.num_beams
        assert tokenized_prompt_completions.shape[0] == batch_size * num_beams
        # add 10 to account for chat template tokens
        assert (
            tokenized_prompt_completions.shape[1]
            <= trainer.max_length + trainer.max_new_tokens + 10
        )

        # Verify prompt_completions (decoded strings)
        assert len(prompt_completions) == batch_size * num_beams
        assert all(isinstance(completion, str) for completion in prompt_completions)

        # Verify that no gradient flows through here
        assert not tokenized_prompt_completions.requires_grad, "No gradients through rollouts"

        # Verify that each completion starts with its corresponding prompt
        for i, prompt in enumerate(prompts):
            for beam_idx in range(num_beams):
                completion_idx = i * num_beams + beam_idx
                completion = prompt_completions[completion_idx]
                # Completion should start with the prompt (accounting for tokenization differences)
                assert prompt in completion, (
                    f"Completion {completion_idx} (beam {beam_idx} for prompt {i}) "
                    f"does not contain prompt '{prompt}'. Got: '{completion}'"
                )

    def test_calculate_advantages(self, trainer):
        """Test _calculate_advantages() with known rewards to verify normalization."""
        # Setup
        batch_size = 2
        num_beams = 3
        prompt_completions0 = [
            "prompt 0: solution=$0$",
            "prompt 0: solution=$0$",
            "prompt 0: solution=$10$",
        ]
        solutions0 = ["0", "0", "0"]
        expected_rewards0 = [1.0, 1.0, 0.0]

        prompt_completions1 = [
            "prompt 1: solution=$10$",
            "prompt 1: solution=$1$",
            "prompt 1: solution=$10$",
        ]
        solutions1 = ["1", "1", "1"]
        expected_rewards1 = [0.0, 1.0, 0.0]

        prompt_completions = prompt_completions0 + prompt_completions1
        solutions = solutions0 + solutions1

        # Mock the reward_fn method to return our specific values
        def mock_reward_fn(completions, solution=None):
            # Return rewards in the same order as completions
            pattern = r"\$(\d+)\$"

            rewards = []
            for completion_item, sol in zip(completions, solution, strict=True):
                completion_text = completion_item[0]["content"]
                match = re.search(pattern, completion_text)
                if match:
                    extracted_number = match.group(1)
                    if extracted_number == sol:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)

            return rewards

        trainer.reward_fn = mock_reward_fn

        # Calculate advantages
        advantages, mean_rewards = trainer._calculate_advantages(
            prompt_completions=prompt_completions,
            solutions=solutions,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        # Verify shape
        assert advantages.shape == (batch_size, num_beams)

        # Calculate expected advantages analytically
        rewards_0 = torch.tensor(expected_rewards0)
        mean_0 = rewards_0.mean()
        std_0 = rewards_0.std()
        expected_advantages_0 = (rewards_0 - mean_0) / std_0

        # For prompt 1: [0.0, 1.0, 2.0]
        rewards_1 = torch.tensor(expected_rewards1)
        mean_1 = rewards_1.mean()
        std_1 = rewards_1.std()
        expected_advantages_1 = (rewards_1 - mean_1) / std_1

        # Verify mean rewards
        assert torch.allclose(mean_rewards[0], mean_0.to(trainer.device), atol=1e-5)
        assert torch.allclose(mean_rewards[1], mean_1.to(trainer.device), atol=1e-5)

        # Verify advantages match expected values
        assert torch.allclose(advantages[0], expected_advantages_0.to(trainer.device), atol=1e-5)
        assert torch.allclose(advantages[1], expected_advantages_1.to(trainer.device), atol=1e-5)

        # Verify that each row has zero mean and unit variance (key property of normalization)
        for i in range(batch_size):
            row_mean = advantages[i].mean()
            row_std = advantages[i].std()
            assert torch.allclose(
                row_mean, torch.tensor(0.0), atol=1e-5
            ), f"Row {i} should have zero mean, got {row_mean}"
            assert torch.allclose(
                row_std, torch.tensor(1.0), atol=1e-5
            ), f"Row {i} should have unit variance, got {row_std}"

    def test_get_attention_mask(self, trainer):
        """Test _get_attention_mask() with focus on prompt masking and EOS handling."""
        batch_size = 2
        num_beams = 2
        prompt_length = 3
        completion_length = 5
        seq_length = prompt_length + completion_length

        # Create tokenized prompt with some padding
        tokenized = {
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1],  # All prompt tokens valid
                    [0, 1, 1],  # Last prompt token is padding
                ],
                device=trainer.device,
                dtype=torch.long,
            )
        }

        # Create output sequences with different EOS scenarios
        eos_token_id = trainer.tokenizer.tokenizer.eos_token_id
        output = torch.tensor(
            [
                [1, 2, 3, 10, 11, eos_token_id, 13, 14],
                [1, 2, 3, 20, 21, 22, 23, 24],
                [4, 5, 6, 30, eos_token_id, 32, eos_token_id, 34],
                [4, 5, 6, 40, 41, 42, 43, eos_token_id],
            ],
            device=trainer.device,
            dtype=torch.long,
        )

        attention_mask = trainer._get_attention_mask(
            batch_size=batch_size,
            num_beams=num_beams,
            seq_length=seq_length,
            prompt_length=prompt_length,
            tokenized=tokenized,
            output=output,
        )

        # 1. Shape is correct
        assert attention_mask.shape == (batch_size * num_beams, seq_length)

        # 2. Prompt tokens are correctly masked from tokenized input
        # Prompt 0: all tokens valid -> should be [1, 1, 1] for both beams
        assert (
            attention_mask[0, :prompt_length] == torch.tensor([1, 1, 1], device=trainer.device)
        ).all()
        assert (
            attention_mask[1, :prompt_length] == torch.tensor([1, 1, 1], device=trainer.device)
        ).all()

        # Prompt 1: last token is padding -> should be [0, 1, 1] for both beams
        assert (
            attention_mask[2, :prompt_length] == torch.tensor([0, 1, 1], device=trainer.device)
        ).all()
        assert (
            attention_mask[3, :prompt_length] == torch.tensor([0, 1, 1], device=trainer.device)
        ).all()

        # 3. EOS handling: everything after first EOS (including EOS) should be masked
        # Prompt 0, Beam 0: EOS at completion position 2 -> positions 0,1,2 valid, 3,4 masked
        assert (
            attention_mask[0, prompt_length : prompt_length + 3] == 1
        ).all(), "Tokens up to and including EOS should be valid"
        assert (
            attention_mask[0, prompt_length + 3 :] == 0
        ).all(), "Tokens after EOS should be masked"

        # Prompt 0, Beam 1: No EOS -> all completion tokens valid
        assert (
            attention_mask[1, prompt_length:] == 1
        ).all(), "All completion tokens should be valid when no EOS"

        # Prompt 1, Beam 0: First EOS at completion position 1 -> positions 0,1 valid, 2,3,4 masked
        # (Multiple EOS tokens, but only first one matters)
        assert (
            attention_mask[2, prompt_length : prompt_length + 2] == 1
        ).all(), "Tokens up to and including first EOS should be valid"
        assert (
            attention_mask[2, prompt_length + 2 :] == 0
        ).all(), "Tokens after first EOS should be masked (even if there are more EOS tokens)"

        # Prompt 1, Beam 1: EOS at last position -> all completion tokens valid (EOS is included)
        assert (
            attention_mask[3, prompt_length:] == 1
        ).all(), "All completion tokens including last EOS should be valid"

    def test_get_log_probs(self, trainer, small_model):
        """Test _get_log_probs() with focus on prompt exclusion and masking."""
        batch_size = 2
        num_beams = 3
        prompt_length = 3
        completion_length = 4

        # Create token sequences: [prompt | completion]
        tokenized_prompt_completions = torch.tensor(
            [
                [1, 2, 3, 10, 11, 12, 13],  # Prompt 0, Beam 0
                [1, 2, 3, 20, 21, 22, 23],  # Prompt 0, Beam 1
                [1, 2, 3, 30, 31, 32, 33],  # Prompt 0, Beam 2
                [4, 5, 6, 40, 41, 42, 43],  # Prompt 1, Beam 0
                [4, 5, 6, 50, 51, 52, 53],  # Prompt 1, Beam 1
                [4, 5, 6, 60, 61, 62, 63],  # Prompt 1, Beam 2
            ],
            device=trainer.device,
            dtype=torch.long,
        )

        # Create mask: some completion tokens are masked (0 = masked)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1],  # All valid
                [1, 1, 1, 1, 1, 0, 0],  # Last 2 completion tokens masked
                [0, 0, 1, 1, 1, 1, 1],  # First 2 prompt tokens are padded
                [1, 1, 1, 1, 1, 1, 0],  # Last completion token masked
                [1, 1, 1, 1, 1, 1, 1],  # All valid
                [0, 1, 1, 1, 1, 1, 0],  # First prompt token padded, and last completion
            ],
            device=trainer.device,
            dtype=torch.long,
        )

        log_probs_summed, log_probs_per_token = trainer._get_log_probs(
            model=small_model,
            prompt_length=prompt_length,
            batch_size=batch_size,
            num_beams=num_beams,
            tokenized_prompt_completions=tokenized_prompt_completions,
            attention_mask=attention_mask,
            use_no_grad=True,
        )

        # Core checks
        # 1. Shapes are correct
        assert log_probs_summed.shape == (batch_size, num_beams)
        assert log_probs_per_token.shape == (batch_size * num_beams, completion_length)

        # 2. Masked tokens don't contribute (should be zero)
        assert (log_probs_per_token[1, -2:] == 0).all(), "Masked tokens should be zero"
        assert log_probs_per_token[3, -1:] == 0, "Masked token should be zero"
        assert log_probs_per_token[5, -1:] == 0, "Masked token should be zero"

        # 3. Summed log probs match sum of unmasked per-token log probs
        assert torch.allclose(log_probs_summed[0, 1], log_probs_per_token[1, :].sum(), atol=1e-5)
        assert torch.allclose(log_probs_summed[1, 0], log_probs_per_token[3, :].sum(), atol=1e-5)

    def test_get_log_probs_gather_correctness(self, trainer):
        """Test that gather correctly extracts log probs for actual generated tokens."""
        batch_size, num_beams, prompt_length, seq_length = 2, 2, 3, 7
        completion_length = seq_length - prompt_length
        vocab_size = 100

        # Token sequences: [prompt | completion]
        tokenized_prompt_completions = torch.tensor(
            [
                [1, 2, 3, 10, 11, 12, 13],  # completion: [10, 11, 12, 13]
                [1, 2, 3, 20, 21, 22, 23],  # completion: [20, 21, 22, 23]
                [4, 5, 6, 30, 31, 32, 33],  # completion: [30, 31, 32, 33]
                [4, 5, 6, 40, 41, 42, 43],  # completion: [40, 41, 42, 43]
            ],
            device=trainer.device,
            dtype=torch.long,
        )

        attention_mask = torch.ones(
            (batch_size * num_beams, seq_length),
            device=trainer.device,
            dtype=torch.long,
        )

        # Mock model: logits[i, j, :] predicts token at position j+1
        mock_model = MagicMock()
        mock_logits = torch.zeros(
            (batch_size * num_beams, seq_length, vocab_size),
            device=trainer.device,
        )

        # Set high logit for correct token at each position
        for i in range(batch_size * num_beams):
            for j in range(prompt_length - 1, seq_length - 1):
                token_id = tokenized_prompt_completions[i, j + 1].item()
                mock_logits[i, j, token_id] = 10.0

        mock_model.forward.return_value = MagicMock(logits=mock_logits)

        log_probs_summed, log_probs_per_token = trainer._get_log_probs(
            model=mock_model,
            prompt_length=prompt_length,
            batch_size=batch_size,
            num_beams=num_beams,
            tokenized_prompt_completions=tokenized_prompt_completions,
            attention_mask=attention_mask,
            use_no_grad=False,
        )

        # Verify gather extracts correct log probs
        for i in range(batch_size * num_beams):
            for j in range(completion_length):
                token_id = tokenized_prompt_completions[i, prompt_length + j].item()
                logit_pos = prompt_length - 1 + j
                expected = torch.nn.functional.log_softmax(mock_logits[i, logit_pos, :], dim=-1)[
                    token_id
                ]
                assert torch.allclose(log_probs_per_token[i, j], expected, atol=1e-5)

        # Verify summed log probs
        assert torch.allclose(
            log_probs_summed,
            log_probs_per_token.view(batch_size, num_beams, completion_length).sum(dim=-1),
            atol=1e-5,
        )

    def test_calculate_kl_divergence(self, trainer):
        """Test _calculate_kl_divergence() using PyTorch's KL implementation as reference."""
        # Create log probabilities
        log_probs_per_token = torch.tensor(
            [
                [math.log(0.5), math.log(0.5), math.log(0.25), math.log(0.5)],
                [math.log(0.5), math.log(0.8), math.log(0.25), math.log(0.2)],
            ],
            device=trainer.device,
            dtype=torch.float32,
        )

        ref_log_probs_per_token = torch.tensor(
            [
                [math.log(0.5), math.log(0.5), math.log(0.25), math.log(0.5)],
                [math.log(0.5), math.log(0.1), math.log(0.5), math.log(0.9)],
            ],
            device=trainer.device,
            dtype=torch.float32,
        )

        # Calculate expected KL div across all sequences and all batches
        expected_kl_per_token = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.8 * math.log(0.8 / 0.1),
                    0.25 * math.log(0.25 / 0.5),
                    0.2 * math.log(0.2 / 0.9),
                ],
            ],
            device=trainer.device,
        )

        kl_div_per_token = trainer._calculate_kl_divergence(
            log_probs_per_token=log_probs_per_token,
            ref_log_probs_per_token=ref_log_probs_per_token,
        )

        # Verify KL divergence matches reference implementation
        assert torch.allclose(kl_div_per_token, expected_kl_per_token, atol=1e-5)

    def test_calculate_grpo_loss(self, trainer):
        """Test _calculate_grpo_loss() with known values to verify clipping and objective calculation."""
        # Create known probabilities
        # Shape: (batch_size * num_beams, completion_length) = (2, 2)
        # [[not clipped, not clipped], [too high, too low]]
        probs = torch.tensor(
            [
                [0.8, 0.6],
                [0.4, 0.5],
            ],
            device=trainer.device,
            dtype=torch.float32,
        )

        ref_probs = torch.tensor(
            [
                [0.8, 0.65],
                [0.3, 0.7],
            ],
            device=trainer.device,
            dtype=torch.float32,
        )

        prob_ratio = torch.tensor(
            [[0.8 / 0.8, 0.6 / 0.65], [0.4 / 0.3, 0.5 / 0.7]], device=trainer.device
        )

        # Advantages (already normalized)
        # Shape: (batch_size, num_beams) = (1, 2)
        advantages = torch.tensor(
            [[0.707, -0.707]],
            device=trainer.device,
            dtype=torch.float32,
        )

        grpo_loss_per_token = trainer._calculate_grpo_loss(
            log_probs_per_token=torch.log(probs),
            ref_log_probs_per_token=torch.log(ref_probs),
            advantages=advantages,
            batch_size=1,
            num_beams=2,
        )

        # Calculate expected result manually
        # eps = 0.1 (from trainer initialization)
        eps = trainer.eps
        clipped_prob_ratio = torch.clamp(prob_ratio, min=1 - eps, max=1 + eps)
        advantages_expanded = advantages.view(1 * 2, 1).expand_as(probs)  # (2, 2)
        expected_grpo_per_token = torch.minimum(
            prob_ratio * advantages_expanded, clipped_prob_ratio * advantages_expanded
        )

        # Verify GRPO loss matches expected calculation
        assert torch.allclose(grpo_loss_per_token, expected_grpo_per_token, atol=1e-5)

        # Verify clipping works: when ratio exceeds bounds, clipped version is used
        assert prob_ratio[0, 1] > 1 - eps, "Ratio should be within bounds"
        assert prob_ratio[0, 1] < 1 + eps, "Ratio should be within bounds"
        assert prob_ratio[0, 1] == clipped_prob_ratio[0, 1], "Ratio should not be clipped"
        assert prob_ratio[1, 0] > 1 + eps, "Ratio should exceed upper bound"
        assert clipped_prob_ratio[1, 0] == 1 + eps, "Ratio should be clipped to upper bound"
        assert prob_ratio[1, 1] < 1 - eps, "Ratio should exceed lower bound"
        assert clipped_prob_ratio[1, 1] == 1 - eps, "Ratio should be clipped to lower bound"
