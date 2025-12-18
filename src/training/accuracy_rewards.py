"""Accuracy reward function matching HuggingFace TRL implementation.

This is a simplified version of trl.rewards.accuracy_reward that removes
HF-specific abstractions while maintaining identical behavior.
"""

try:
    from latex2sympy2_extended import NormalizationConfig  # type: ignore[import-untyped]
    from math_verify import LatexExtractionConfig, parse, verify  # type: ignore[import-untyped]

    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    _MATH_VERIFY_AVAILABLE = False


def accuracy_reward(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[float | None]:
    r"""
    Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    Example:
    ```python
    >>> from training.accuracy_rewards import accuracy_reward

    >>> solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
    >>> completions = [
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{3}}"}],
    ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{2}}"}],
    ... ]
    >>> accuracy_reward(completions, solutions)
    [1.0, 0.0]
    ```
    """
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError("Please install the `math_verify` package to use accuracy_reward")

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution, strict=True):
        gold_parsed = parse(sol)
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = float(verify(gold_parsed, answer_parsed))
        else:
            # If the gold solution cannot be parsed, we assign `None` to skip this example
            reward = None
        rewards.append(reward)

    return rewards
