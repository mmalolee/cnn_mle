import argparse
import torch


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNN model training params")

    # Training config args (child class)
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size for Data Loader")
    parser.add_argument("--model_name", type=str, help="Output model name")

    # Inference config args (parent class)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cuda, cpu, mps)",
    )

    return parser.parse_args()
