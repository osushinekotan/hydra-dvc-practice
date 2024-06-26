import hydra
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def main() -> None:
    """Preprocess the raw data."""
    cfg = OmegaConf.load("params.yaml")
    logger.info("Preprocessing the raw data...")
    logger.info(f"Scaler: {cfg.scaler._target_}")
    train_df = pd.read_csv(f"{cfg.paths.data_dir}/train.csv")
    test_df = pd.read_csv(f"{cfg.paths.data_dir}/test.csv")

    # scaling
    scaler = hydra.utils.instantiate(cfg.scaler)
    scaler.fit(train_df.drop(columns=[cfg.target]))

    train_df_scaled = pd.concat(
        [
            pd.DataFrame(
                scaler.transform(train_df.drop(columns=[cfg.target])),
                columns=train_df.drop(columns=[cfg.target]).columns,
            ),
            train_df[[cfg.target]],
        ],
        axis=1,
    )
    test_df_scaled = pd.concat(
        [
            pd.DataFrame(
                scaler.transform(test_df.drop(columns=[cfg.target])),
                columns=test_df.drop(columns=[cfg.target]).columns,
            ),
            test_df[[cfg.target]],
        ],
        axis=1,
    )

    train_df_scaled.to_csv(f"{cfg.paths.data_dir}/train_scaled.csv", index=False)
    test_df_scaled.to_csv(f"{cfg.paths.data_dir}/test_scaled.csv", index=False)

    logger.info(f"Train set shape: {train_df_scaled.shape}")
    logger.info(f"Test set shape: {test_df_scaled.shape}")

    logger.info("Completed preprocessing the raw data 🎉")


if __name__ == "__main__":
    main()
