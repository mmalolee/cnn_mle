from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig
from src.utils import get_logger
from src.parser import parse_args
from src.data_manager import LoaderData
from src.architectures.cnn import CNN
from src.trainer import ModelTrainer

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting pipeline.")

    try:
        args = parse_args()

        logger.info("Loading configurations.")
        p_cfg = PathsConfig()
        t_cfg = TrainingConfig(
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

        logger.info("Initializing Data Manager.")
        data_manager = LoaderData(path_config=p_cfg, train_config=t_cfg)
        train_loader = data_manager.get_training_data_loader()

        logger.info("Building model")
        model = CNN(model_cfg=t_cfg.model_cfg)

        logger.info("Initializing Model Trainer.")
        trainer = ModelTrainer(model=model, config=t_cfg, paths=p_cfg)

        trainer.fit(train_loader=train_loader)

        logger.info("Pipeline execution finished successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed at root level: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
