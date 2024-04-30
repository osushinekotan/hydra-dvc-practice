import logging

import pandas as pd
from omegaconf import OmegaConf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    """Load the breast cancer dataset and split it into train and test sets."""
    cfg = OmegaConf.load("conf/params.yaml")
    logger.info("Loading and splitting the dataset...")
    feature_names = load_breast_cancer()["feature_names"]
    raw_df = pd.concat(
        [
            pd.DataFrame(load_breast_cancer()["data"], columns=feature_names),
            pd.DataFrame(load_breast_cancer()["target"], columns=["target"]),
        ],
        axis=1,
    )
    train_df, test_df = train_test_split(raw_df, test_size=cfg.test_size, random_state=cfg.seed)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(f"{cfg.paths.data_dir}/train.csv", index=False)
    test_df.to_csv(f"{cfg.paths.data_dir}/test.csv", index=False)

    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")
    logger.info("Completed loading and splitting the dataset 🎉")


if __name__ == "__main__":
    main()
