import json
import logging

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train a model."""
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
