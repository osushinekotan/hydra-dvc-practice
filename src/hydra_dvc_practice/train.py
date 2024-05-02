import json

import hydra
import joblib
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def main() -> None:
    """Train a model."""
    cfg = OmegaConf.load("params.yaml")
    logger.info("Training a model... (no validation for simplicity)")
    logger.info(f"Model: {cfg.model._target_}")
    train_df = pd.read_csv(f"{cfg.paths.data_dir}/train_scaled.csv")

    X_train = train_df.drop(columns=[cfg.target])
    y_train = train_df[cfg.target]
    model = hydra.utils.instantiate(cfg.model)
    model.fit(X_train, y_train)

    # save model params
    params = model.get_params()
    logger.info(f"Model params:\n{params}")
    with open(f"{cfg.paths.model_dir}/model_params.json", "w") as f:
        json.dump(params, f)

    joblib.dump(model, f"{cfg.paths.model_dir}/model.pkl")
    logger.info("Completed training a model 🎉")


if __name__ == "__main__":
    main()
