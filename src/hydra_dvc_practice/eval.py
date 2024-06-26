import json

import joblib
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main() -> None:
    """Make predictions using the trained model."""
    cfg = OmegaConf.load("params.yaml")
    logger.info("Making predictions...")
    test_df = pd.read_csv(f"{cfg.paths.data_dir}/test_scaled.csv")
    model = joblib.load(f"{cfg.paths.model_dir}/model.pkl")

    X_test = test_df.drop(columns=[cfg.target])
    y_test = test_df[cfg.target]
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score,
        "f1": f1_score,
        "recall": recall_score,
        "precision": precision_score,
    }
    eval_results = {}
    for metric_name, metric_fn in metrics.items():
        metric_value = metric_fn(y_test, y_pred)
        logger.info(f"{metric_name.capitalize()}: {metric_value:.2f}")
        eval_results[metric_name] = metric_value

    # save evaluation results
    with open(f"{cfg.paths.model_dir}/evaluation_results.json", "w") as f:
        json.dump(eval_results, f)

    # save predictions
    test_df["prediction"] = y_pred
    test_df.to_csv(f"{cfg.paths.data_dir}/test_predictions.csv", index=False)

    logger.info(f"Test set shape: {test_df.shape}")
    logger.info("Completed making predictions 🎉")


if __name__ == "__main__":
    main()
