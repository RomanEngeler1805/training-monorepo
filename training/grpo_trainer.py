import time

import matplotlib.pyplot as plt
import torch

from data.dataloader import DataLoader
from inference.decoding import BeamDecoder
from models.transformer import Model, ScratchModel, Tokenizer
from models.transformer import Model as HFModel
from training.optimizer import SGD
from utils.utils import logger

# Resource: https://huggingface.co/docs/trl/main/en/grpo_trainer


class GRPOTrainer:
    def __init__(
        self,
        model: HFModel | ScratchModel,
        ref_model: HFModel | ScratchModel,
        tokenizer: Tokenizer,
        dataloader: DataLoader,
        max_length: int,
        max_new_tokens: int,
        eps: float,
        beta: float,
        optimizer: torch.optim.Optimizer | SGD,
        decoder: BeamDecoder,
        reward_fn,
        max_grad_norm: float = 10.0,
    ):
        """Initialize GRPO (Group Relative Policy Optimization) trainer.

        Args:
            model: The policy model to be trained (current model).
            ref_model: Reference model (frozen copy) for KL divergence regularization.
            tokenizer: Tokenizer for encoding/decoding text.
            dataloader: DataLoader providing training batches.
            max_length: Maximum total sequence length for tokenization.
            max_new_tokens: Maximum number of new tokens to generate.
            eps: Clipping parameter for probability ratio (typically 0.1-0.2).
            beta: KL divergence penalty weight (typically 0.01-0.1).
            optimizer: Optimizer for updating model parameters.
            decoder: Decoder strategy for generation (e.g., BeamDecoder with num_beams=K).
            max_grad_norm: Maximum gradient norm for gradient clipping.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.decoder = decoder
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.eps = eps
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.reward_fn = reward_fn

        self.ref_model = ref_model
        self.ref_model.eval()  # Freeze it
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.device = next(self.model.parameters()).device

    def _generate_completions(
        self, prompts: list[str]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str]]:
        """Generate K completions per prompt using beam search.

        Args:
             prompts: List of prompt strings (batch_size).

        Returns:
            Tuple of:
            - tokenized_prompt: Tokenized prompts dictionary with "input_ids" and "attention_mask" (batch_size, prompt_length).
            - tokenized_prompt_completions: Generated token IDs of shape (batch_size * num_beams, seq_length).
            - prompt_completions: List of decoded completion strings (length: batch_size * num_beams).
        """
        # Debug: Print original sample from dataloader (no template)
        logger.debug(f"   Original prompt: {prompts[0][:500]}...")

        # Tokenize prompts with chat template
        tokenized_prompt = self.tokenizer.tokenize(
            input=prompts, max_length=self.max_length, apply_chat_template=True
        )

        # Debug: Verify template was added by decoding
        decoded_with_template = self.tokenizer.tokenizer.decode(
            tokenized_prompt["input_ids"][0], skip_special_tokens=False
        )
        logger.debug(f"   Detokenized prompt: {decoded_with_template[:500]}...")

        # Move tensors to device (tokenize returns a dict)
        tokenized_prompt = {k: v.to(device=self.device) for k, v in tokenized_prompt.items()}

        # Generate K completions per prompt using beam search
        # Note: model.generate() sets model to eval / inference mode internally
        tokenized_prompt_completions = self.model.generate(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            decoder=self.decoder,
            num_beams=self.decoder.num_beams,
        )

        # Clone to create a new tensor that can be used in autograd
        # The original was created in inference_mode() and can't be used for backward
        tokenized_prompt_completions = tokenized_prompt_completions.clone().detach()
        tokenized_prompt_completions.requires_grad_(False)

        # Decode generated sequences back to text for reward calculation
        prompt_completions = self.tokenizer.batch_decode(token_ids=tokenized_prompt_completions)

        # Debug: Print one generation to verify it works
        logger.debug(f"   Generated completion: {prompt_completions[0]}")

        return tokenized_prompt, tokenized_prompt_completions, prompt_completions

    def _calculate_advantages(
        self,
        prompt_completions: list[str],
        solutions: list[str],
        batch_size: int,
        num_beams: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate group-relative advantages for GRPO objective.

        Normalizes rewards within each prompt group to zero mean and unit variance.
        advantage = (reward - mean(rewards)) / std(rewards)

        Args:
            prompt_completions: List of completion strings (length: batch_size * num_beams).
                              Completions are ordered as [prompt0_beam0, prompt0_beam1, ...,
                              prompt1_beam0, prompt1_beam1, ...].
            solutions: List of solution strings, length (batch_size * num_beams)
            batch_size: Number of prompts in the batch.
            num_beams: Number of completions per prompt

        Returns:
            Tuple of:
            - advantages: Shape (batch_size, num_beams), zero mean and unit variance per row.
            - mean_rewards: Shape (batch_size,), mean reward per prompt.
        """
        # Compute rewards for all completions
        # Note: reward() expects list[list[dict[str, str]]] format
        raw_rewards = self.reward_fn(
            completions=[[{"content": pc}] for pc in prompt_completions],
            solution=solutions,  # Pass prompts for solution lookup
        )
        logger.debug(f"   Raw rewards {raw_rewards}")
        # Replace None values with 0.0 (happens when gold solution can't be parsed)
        raw_rewards = [r if r is not None else 0.0 for r in raw_rewards]
        rewards = torch.tensor(data=raw_rewards, device=self.device, dtype=torch.float32)

        # Reshape to group rewards by prompt: (batch_size * num_beams,) -> (batch_size, num_beams)
        rewards = rewards.view(batch_size, num_beams)

        # Calculate mean reward per prompt (before normalization)
        mean_rewards = torch.mean(rewards, dim=-1)  # (batch_size,)

        # Normalize within each group (prompt) to zero mean and unit variance
        rewards_mean = torch.mean(rewards, dim=-1, keepdim=True)  # (batch_size, 1)
        rewards_std = torch.std(rewards, dim=-1, keepdim=True)  # (batch_size, 1)
        rewards_std = torch.clamp(rewards_std, min=1e-8)  # Prevent division by zero

        # Compute normalized advantages: (rewards - mean) / std
        advantages = (rewards - rewards_mean) / rewards_std

        return advantages, mean_rewards

    def _get_attention_mask(
        self,
        batch_size: int,
        num_beams: int,
        seq_length: int,
        prompt_length: int,
        tokenized: dict[str, torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Create attention mask for full sequences (prompt + completion).

        Creates a mask that:
        1. Includes all valid prompt tokens (from tokenizer attention mask)
        2. Includes completion tokens up to and including the first EOS token
        3. Masks out padding/EOS tokens after the first EOS

        Args:
            batch_size: Number of prompts in the batch.
            num_beams: Number of completions per prompt.
            seq_length: Total sequence length (prompt + completion).
            prompt_length: Length of prompt tokens.
            tokenized: Dictionary containing tokenized inputs with "attention_mask" key.
            output: Generated token IDs of shape (batch_size * num_beams, seq_length).

        Returns:
            Attention mask tensor of shape (batch_size * num_beams, seq_length) where
            1 indicates valid tokens and 0 indicates padding/masked tokens.
        """
        # Create attention mask
        attention_mask = torch.zeros(
            batch_size * num_beams, seq_length, device=self.device, dtype=torch.long
        )

        # Mask out padding tokens in prompt (repeat mask for all beams)
        prompt_attention_mask = tokenized["attention_mask"].repeat_interleave(num_beams, dim=0)
        attention_mask[:, :prompt_length] = prompt_attention_mask

        # Mask out padding tokens in completion
        eos_token_id = self.tokenizer.tokenizer.eos_token_id
        completions = output[:, prompt_length:]  # (batch_size * num_beams, completion_length)
        is_eos = completions == eos_token_id
        first_eos_pos = is_eos.long().argmax(dim=1)  # First EOS index, or 0 if none found
        has_eos = is_eos.any(dim=1)
        # Include EOS token: +1, or use full length if no EOS
        first_eos_pos = torch.where(has_eos, first_eos_pos + 1, completions.shape[1])

        # Vectorized mask: positions < first_eos_pos (includes EOS since we added +1)
        completion_length = completions.shape[1]
        positions = torch.arange(completion_length, device=self.device)
        attention_mask[:, prompt_length:] = (
            positions.unsqueeze(0) < first_eos_pos.unsqueeze(1)
        ).long()
        return attention_mask

    def _get_log_probs(
        self,
        model: torch.nn.Module | ScratchModel | Model,
        prompt_length: int,
        batch_size: int,
        num_beams: int,
        tokenized_prompt_completions: torch.Tensor,
        attention_mask: torch.Tensor,
        use_no_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract log-probabilities of generated completions under the model.

        Computes log-probability of each generated token, masks out invalid tokens
        (padding after EOS), and sums to get log-probability of entire completion.

        Args:
            model: Model to compute log-probabilities under
            prompt_length: Length of prompt tokens
            batch_size: Number of prompts in the batch.
            num_beams: Number of completions per prompt.
            tokenized_prompt_completions: Prompt and generation tokens of shape (batch_size * num_beams, seq_length).
            attention_mask: Prompt attention mask of shape (batch_size * num_beams, seq_length).
            use_no_grad: If True, disable gradients (for reference model). If False, keep gradients (for current model).

        Returns:
            Tuple of (summed_log_probs, per_token_log_probs) where:
            - summed_log_probs: shape (batch_size, num_beams) containing log-probability of each completion
            - per_token_log_probs: shape (batch_size * num_beams, completion_length) containing per-token log-probabilities
        """
        # 1. Feed (prompt + completion) through model to get logits
        #    Shape: (batch_size * num_beams, seq_length, vocab_size)
        if use_no_grad:
            with torch.no_grad():
                model_logits = model.forward(
                    input_ids=tokenized_prompt_completions, attention_mask=attention_mask
                ).logits
        else:
            model_logits = model.forward(
                input_ids=tokenized_prompt_completions, attention_mask=attention_mask
            ).logits

        # 2. Extract log-probs for actually generated completion tokens
        # (batch_size * num_beams, completion_length, vocab_size)
        completion_logits = model_logits[:, prompt_length - 1 : -1, :]
        # (batch_size * num_beams, completion_length)
        completion_token_ids = tokenized_prompt_completions[:, prompt_length:]

        # Get log-probs for the actual generated tokens using gather
        # (batch_size * num_beams, completion_length, vocab_size)
        log_probs_all_vocab = torch.nn.functional.log_softmax(completion_logits, dim=-1)
        # (batch_size * num_beams, completion_length)
        log_probs_per_token = torch.gather(
            log_probs_all_vocab, dim=-1, index=completion_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        # 3. Mask out invalid tokens (prompt part already excluded, mask post-EOS tokens)
        # (batch_size * num_beams, completion_length)
        completion_mask = attention_mask[:, prompt_length:]
        log_probs_per_token = log_probs_per_token * completion_mask.float()

        # Sum over sequence to get log-prob of each completion
        # (batch_size * num_beams,)
        log_probs_summed = log_probs_per_token.sum(dim=-1)

        # Reshape for advantage calculation
        log_probs_summed = log_probs_summed.view(batch_size, num_beams)

        return log_probs_summed, log_probs_per_token

    def _calculate_kl_divergence(
        self,
        log_probs_per_token: torch.Tensor,
        ref_log_probs_per_token: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence between current and reference model distributions.

        Computes KL(current || reference) = p_current * (log(p_current) - log(p_reference))
        for each generated token.

        Args:
            log_probs_per_token: Log-probabilities of generated tokens under current model.
                                Shape: (batch_size * num_beams, completion_length).
            ref_log_probs_per_token: Log-probabilities of generated tokens under reference model.
                                    Shape: (batch_size * num_beams, completion_length).

        Returns:
            Per token KL divergence. Shape: (batch_size * num_beams, completion_length).
        """
        probs_generated = torch.exp(log_probs_per_token)
        kl_per_token = probs_generated * (log_probs_per_token - ref_log_probs_per_token)

        return kl_per_token

    def _calculate_grpo_loss(
        self,
        log_probs_per_token: torch.Tensor,
        ref_log_probs_per_token: torch.Tensor,
        advantages: torch.Tensor,
        batch_size: int,
        num_beams: int,
    ) -> torch.Tensor:
        """Calculate GRPO objective loss with probability ratio clipping per token.

        Computes the GRPO objective per token: min(ratio * advantage, clipped_ratio * advantage)
        where the probability ratio is clipped to [1-eps, 1+eps] to prevent large policy updates.
        Advantages are per-sequence (outcome rewards), so they are broadcast to each token in the sequence.
        Loss is calculated per token, then summed over tokens in each sequence, then averaged over batch.

        Args:
            log_probs_per_token: Log-probabilities of generated tokens under current model.
                               Shape: (batch_size * num_beams, completion_length).
            ref_log_probs_per_token: Log-probabilities of generated tokens under reference model.
                                    Shape: (batch_size * num_beams, completion_length).
            advantages: Group-relative advantages (per-sequence). Shape: (batch_size, num_beams).
            batch_size: Number of prompts in the batch.
            num_beams: Number of completions per prompt.

        Returns:
            Per token GRPO objective. Shape: (batch_size * num_beams, completion_length).
        """
        # Calculate per-token log probability ratio
        # Shape: (batch_size * num_beams, completion_length)
        log_prob_ratio_per_token = log_probs_per_token - ref_log_probs_per_token
        prob_ratio_per_token = torch.exp(log_prob_ratio_per_token)  # π_θ / π_ref
        clipped_prob_ratio_per_token = torch.clamp(
            prob_ratio_per_token, min=1 - self.eps, max=1 + self.eps
        )

        # Broadcast advantages from (batch_size, num_beams) to (batch_size * num_beams, completion_length)
        advantages_expanded = advantages.view(batch_size * num_beams, 1)
        advantages_per_token = advantages_expanded.expand_as(log_probs_per_token)

        # Calculate GRPO objective per token: min(ratio * advantage, clipped_ratio * advantage)
        # Shape: (batch_size * num_beams, completion_length)
        grpo_loss_per_token = torch.minimum(
            prob_ratio_per_token * advantages_per_token,
            clipped_prob_ratio_per_token * advantages_per_token,
        )

        return grpo_loss_per_token

    def _optimization_step(self, loss: torch.Tensor) -> None:
        """Perform one optimization step (backward pass, gradient clipping, parameter update).

        Args:
            loss: Loss tensor to backpropagate.
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

    def train(
        self,
        n_epochs: int = 3,
        plot_losses: bool = True,
        log_interval: int = 5,
    ) -> tuple[list[float], list[float]]:
        """Train the model using GRPO (Group Relative Policy Optimization).

        Training loop that:
        1. Generates K completions per prompt using beam search
        2. Computes rewards and group-relative advantages
        3. Computes GRPO objective with probability ratio clipping
        4. Adds KL divergence regularizer
        5. Updates model parameters

        Args:
            n_epochs: Number of training epochs.
            plot_losses: Whether to plot training losses at the end.
            log_interval: Log loss every N batches (for debug logging).

        Returns:
            Tuple of (losses, rewards) where each is a list of values per batch.
        """
        logger.info("Starting GRPO training...")
        self.model.train()
        losses = []
        rewards = []

        for epoch in range(n_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{n_epochs}")
            epoch_losses = []
            epoch_rewards = []

            # loop through the data loader
            for batch_idx, batch in enumerate(self.dataloader):
                logger.debug(f"Got batch {batch_idx}, size: {len(batch)}")
                batch_size = len(batch)
                num_beams = self.decoder.num_beams
                prompts = [item["prompt"] for item in batch]
                solutions = [item["solution"] for item in batch]
                # Expand solutions to match completions: each solution repeated num_beams times
                # Completions order: [prompt0_beam0, prompt0_beam1, ..., prompt0_beamK-1, prompt1_beam0, ...]
                # Solutions should be: [sol0, sol0, ..., sol0 (K times), sol1, sol1, ..., sol1 (K times), ...]
                solutions = [sol for sol in solutions for _ in range(num_beams)]

                # TODO: test for batch_size > 1

                # Sample K completions per prompt
                logger.debug(
                    f"Batch {batch_idx}: Generating {len(batch)} prompts with {num_beams} beams each..."
                )
                step_start = time.time()

                tokenized_prompt, tokenized_rompt_completions, prompt_completions = (
                    self._generate_completions(prompts)
                )
                logger.debug(f"  Generation: {time.time() - step_start:.3f}s")

                # Calculate group relative advantages (pass batch for solution lookup)
                step_start = time.time()
                advantages, mean_rewards = self._calculate_advantages(
                    prompt_completions, solutions, batch_size, num_beams
                )
                # Track mean reward per batch (average across prompts in batch)
                batch_mean_reward = mean_rewards.mean().item()
                epoch_rewards.append(batch_mean_reward)
                rewards.append(batch_mean_reward)
                logger.debug(f"  Advantage calculation: {time.time() - step_start:.3f}s")

                # Get attention mask
                step_start = time.time()
                seq_length = tokenized_rompt_completions.shape[1]
                prompt_length = tokenized_prompt["input_ids"].shape[1]

                attention_mask = self._get_attention_mask(
                    batch_size,
                    num_beams,
                    seq_length,
                    prompt_length,
                    tokenized_prompt,
                    tokenized_rompt_completions,
                )
                logger.debug(f"  Attention mask creation: {time.time() - step_start:.3f}s")

                # Get log probs (need per-token for KL divergence)
                step_start = time.time()
                log_probs, log_probs_per_token = self._get_log_probs(
                    self.model,
                    prompt_length,
                    batch_size,
                    num_beams,
                    tokenized_rompt_completions,
                    attention_mask,
                )
                logger.debug(f"  Policy log probs: {time.time() - step_start:.3f}s")

                step_start = time.time()
                ref_log_probs, ref_log_probs_per_token = self._get_log_probs(
                    self.ref_model,
                    prompt_length,
                    batch_size,
                    num_beams,
                    tokenized_rompt_completions,
                    attention_mask,
                    use_no_grad=True,
                )
                logger.debug(f"  Reference log probs: {time.time() - step_start:.3f}s")

                # GRPO objective (sequence-level)
                step_start = time.time()
                grpo_loss_per_token = self._calculate_grpo_loss(
                    log_probs_per_token=log_probs_per_token,
                    ref_log_probs_per_token=ref_log_probs_per_token,
                    advantages=advantages,
                    batch_size=batch_size,
                    num_beams=num_beams,
                )
                logger.debug(f"  GRPO loss calculation: {time.time() - step_start:.3f}s")

                # Get KL divergence (per-token, requires per-token log-probs)
                step_start = time.time()
                kl_div_per_token = self._calculate_kl_divergence(
                    log_probs_per_token=log_probs_per_token,
                    ref_log_probs_per_token=ref_log_probs_per_token,
                )
                logger.debug(f"  KL divergence calculation: {time.time() - step_start:.3f}s")

                # Total Loss: -GRPO_objective + beta * KL_divergence
                # We negate GRPO because it's a reward to maximize, but loss should be minimized
                completion_mask = attention_mask[
                    :, prompt_length:
                ]  # (batch_size * num_beams, completion_length)
                step_start = time.time()
                loss_per_token = (
                    -(grpo_loss_per_token - self.beta * kl_div_per_token) * completion_mask.float()
                )

                # Dimension-Reduced GRPO: normalize by constant factor instead of actual token count
                # This removes response length bias as shown in "Understanding R1-Zero-Like Training: A Critical Perspective"
                # Formula: loss = (per_token_loss * completion_mask).sum() / (batch_size * num_beams * max_completion_length)
                # This ensures all sequences contribute equally regardless of their actual length
                num_total_positions = batch_size * num_beams * self.max_new_tokens
                loss = loss_per_token.sum() / num_total_positions
                logger.debug(f"  Loss combination: {time.time() - step_start:.3f}s")

                # Optimization step
                step_start = time.time()
                self._optimization_step(loss)
                logger.debug(f"  Optimization step: {time.time() - step_start:.3f}s")

                loss_value = loss.detach().cpu().item()
                if batch_idx % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_value:.4f}, Reward: {batch_mean_reward:.4f}"
                    )
                    logger.info(
                        f"Completion: {prompt_completions[0][: self.max_new_tokens * 4]}..."
                    )

                epoch_losses.append(loss_value)
                losses.append(loss_value)

                # Cleanup
                del (
                    tokenized_rompt_completions,
                    prompt_completions,
                    attention_mask,
                    loss,
                    tokenized_prompt,
                    log_probs,
                    log_probs_per_token,
                    ref_log_probs,
                    ref_log_probs_per_token,
                    advantages,
                    grpo_loss_per_token,
                    kl_div_per_token,
                    mean_rewards,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            logger.info(
                f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}"
            )

        logger.info("Training completed")

        if plot_losses:
            self._plot_losses(losses, rewards)

        return losses, rewards

    def _plot_losses(self, losses: list[float], rewards: list[float]) -> None:
        """Plot training loss and reward curves.

        Args:
            losses: List of loss values (one per batch) to plot.
            rewards: List of reward values (one per batch) to plot.
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot loss on left y-axis
        color = "tab:red"
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss", color=color)
        ax1.plot(losses, color=color, label="Loss")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Plot reward on right y-axis
        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Reward", color=color)
        ax2.plot(rewards, color=color, label="Reward")
        ax2.tick_params(axis="y", labelcolor=color)

        # Add title
        plt.title("Training Loss and Reward")

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        plt.show()
