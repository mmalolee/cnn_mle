from pathlib import Path


def create_project_structure():
    PROJECT_STRUCTURE = {
        "": [".gitignore", "main.py", "requirements.txt"],
        "data": [],
        "models/checkpoints": ["__init__.py"],
        "src": ["__init__.py", "model_loader.py", "utils.py", "data_manager.py"],
        "src/architectures": ["__init__.py", "cnn.py"],
        "src/cfg": [
            "__init__.py",
            "inference_config.py",
            "model_config.py",
            "paths_config.py",
            "training_config.py",
        ],
        "tests": [],
    }

    for folder, files in PROJECT_STRUCTURE.items():
        folder_path = Path(folder)

        if folder:
            folder_path.mkdir(parents=True, exist_ok=True)

        for file in files:
            (folder_path / file).touch()


if __name__ == "__main__":
    create_project_structure()
