# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward


def main():
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    # Check dataset size and select valid range
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} samples")

    # Select first 15 samples (indices 0-14, which is 1-15 in 1-indexed)
    n_samples = min(512, dataset_size)
    dataset = dataset.select(range(n_samples))
    print(f"Selected {n_samples} samples for training")

    trainer = GRPOTrainer(
        model="google/gemma-3-270m-it",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
