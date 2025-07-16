import os
import sys

pardir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(pardir)
import hydra
from omegaconf import DictConfig

from src.data.utils import load_and_prepare_medmcqa


@hydra.main(config_path="../conf", config_name="dataset", version_base="1.3")
def main(cfg: DictConfig):
    try:
        # Pass dataset path from config
        load_and_prepare_medmcqa(
            max_examples=cfg.max_examples,
            output_path=cfg.dataset_path,
            split_ratio=cfg.split_ratio,
        )
        print("Dataset ready for training!")

    except Exception as e:
        print(f"Failed to prepare dataset: {e}")


if __name__ == "__main__":
    main()
