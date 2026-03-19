from src.data_manager import LoaderData
from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig
from src.utils import get_logger
import torch.nn as nn
import torch

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self, model: nn.Module, config: TrainingConfig, paths: PathsConfig
    ) -> None:
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir / config.exper_name
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.paths = paths
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.cost_func = nn.CrossEntropyLoss()

        logger.info(f"Model moved to {self.device}")
        logger.info(f"ModelTrainer initialized. Training will run on: {self.device}")

    def _save_checkpoint(self, epoch: int, avg_loss: float) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"model_ep_{epoch}.pth"

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": avg_loss,
            "config": self.config,
        }

        torch.save(state, checkpoint_path)

        logger.info(
            f"Checkpoint saved for epoch {epoch} at {self.config.checkpoint_dir}"
        )

    def _train_one_epoch(self, train_loader: LoaderData) -> float:
        self.model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.cost_func(outputs, labels)

            if torch.isnan(loss):
                logger.critical("Loss is NaN. Training has been stopped.")
                raise ValueError("NaN loss.")

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def fit(self, train_loader: LoaderData) -> None:
        logger.info(f"Starting training process for {self.config.epochs} epochs.")

        try:
            for epoch in range(1, self.config.epochs + 1):
                avg_loss = self._train_one_epoch(train_loader)
                logger.info(
                    f"Epoch [{epoch}/{self.config.epochs}] finished. Average Loss: {avg_loss:.4f}"
                )

                self._save_checkpoint(epoch, avg_loss)

            logger.info("Training cycle completed successfully.")

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")

        except Exception as e:
            logger.critical(
                f"Unexpected error during training: {str(e)}", exc_info=True
            )
            raise e
