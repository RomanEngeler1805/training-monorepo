import copy

import torch

from data.dataloader import DataLoader
from data.dataset import Dataset
from inference.decoding import BeamDecoder
from models.transformer import Model, Tokenizer
from training.accuracy_rewards import accuracy_reward
from training.grpo_trainer import GRPOTrainer
from training.optimizer import SGD
from utils.utils import logger


def main():
    # Training hyperparameters
    n_epochs = 20
    # n_samples = 1
    batch_size = 1
    lr = 1e-4  # 2e-2
    lr_decay = 1.0
    max_grad_norm = 10.0

    # GRPO hyperparameters
    eps = 0.1
    beta = 0.0
    num_beams = 8

    # Model hyperparameters
    max_length = 512  # Maximum total sequence length (tokenizer)
    max_new_tokens = 256  # Maximum number of new tokens to generate
    dtype = torch.bfloat16

    # initialise the data loader
    logger.info("Loading data...")
    # dataset = Dataset(data_path="roneneldan/TinyStories", split="train", text_column="text") # SFT dataset
    dataset = Dataset(
        data_path="trl-lib/DeepMath-103K",
        split="train",
        text_column="prompt",
        solution_column="solution",
    )  # GRPO dataset
    dataset.data = dataset.data.select([8])  # .select(range(n_samples))
    from datasets import Dataset as HFDataset

    simple_data = HFDataset.from_dict(
        {
            "prompt": [r"Solve for x $\ frac{2x-5}{3}+4=x+1$", "Solve for x $3(x-2)=12$"],
            "solution": ["$4$", "$6$"],
        }
    )
    dataset.data = simple_data
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Loaded {len(dataset)} samples")

    # initialise the tokenizer & model
    logger.info("Loading model and tokenizer...")
    tokenizer = Tokenizer(tokenizer_name="google/gemma-3-270m-it")
    model = Model(model_name="google/gemma-3-270m-it", dtype=dtype)
    # model = ScratchModel(
    #     n_layers=n_layers,
    #     n_vocab=tokenizer.tokenizer.vocab_size,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     d_hidden=d_hidden,
    #     dropout=dropout,
    #     dtype=dtype,
    # )
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    num_params_millions = num_params / 1_000_000
    logger.info(f"Model ready for training - Parameters: {num_params_millions:.2f}M")

    # optimizer
    optimizer = SGD(model_parameters=model.parameters(), lr=lr, lr_decay=lr_decay)
    logger.info(f"Starting training: {n_epochs} epochs, batch_size={batch_size}, lr={lr}")

    # SFT Trainer
    # loss_fn = CrossEntropy()
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     dataloader=dataloader,
    #     max_length=max_length,
    #     max_grad_norm=max_grad_norm,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    # )

    # GRPO Trainer
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    decoder = BeamDecoder(num_beams=num_beams)

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        eps=eps,
        beta=beta,
        optimizer=optimizer,
        decoder=decoder,
        reward_fn=accuracy_reward,
        max_grad_norm=max_grad_norm,
    )

    trainer.train(n_epochs=n_epochs)


if __name__ == "__main__":
    main()
