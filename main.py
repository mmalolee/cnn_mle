from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig
from src.parser import parse_args
from src.data_manager import LoaderData
from src.architectures.cnn import CNN
from src.trainer import ModelTrainer


def main() -> None:
    args = parse_args()

    p_cfg = PathsConfig()
    t_cfg = TrainingConfig(
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    data_manager = LoaderData(path_config=p_cfg, train_config=t_cfg)
    train_loader = data_manager.get_training_data_loader()

    model = CNN(model_cfg=t_cfg.model_cfg)

    trainer = ModelTrainer(model=model, config=t_cfg, paths=p_cfg)

    try:
        trainer.fit(train_loader=train_loader)

    except KeyboardInterrupt:
        print("zatrzymane z łapy")


if __name__ == "__main__":
    main()
