from src.data_manager import LoaderData
from src.cfg.model_config import ModelConfig
from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig
from src.architectures.cnn import CNN
import torch.nn as nn
import torch


class ModelTraining:
    def __init__(self, model: nn.Module, config: TrainingConfig, paths: PathsConfig):
        self.model = model
        self.config = config
        self.paths = paths
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.cost_func = nn.CrossEntropyLoss()
        self.checkopoint_dir = config.checkpoint_dir

    def _save_checkpoint(self, epoch: int, avg_loss: float):
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.checkpoint_dir / f"model_ep_{epoch}.pth"

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict,
            "loss": avg_loss,
            "config": self.config,
        }

        torch.save(state, checkpoint_path)

    def _train_one_epoch(self, train_loader: LoaderData) -> float:
        self.model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.cost_func(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def fit(self, train_loader: LoaderData):
        print("training init")

        for epoch in range(1, self.config.epochs + 1):
            avg_loss = self._train_one_epoch(train_loader)
            self._save_checkpoint(epoch, avg_loss)

        print("training ended")
