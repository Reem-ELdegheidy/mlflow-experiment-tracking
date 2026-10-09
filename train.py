import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

RANDOM_STATE = 42
EXPERIMENT_NAME = "California_Housing_GBR"


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading California Housing dataset...")

    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    configs = [
        {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
        {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
        {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01}
    ]

    for config in configs:
        print(f"\nTraining {config['run_name']}...")

        with mlflow.start_run(run_name=config["run_name"]):
            model = GradientBoostingRegressor(
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                random_state=RANDOM_STATE
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)

            mlflow.log_params({
                "max_depth": config["max_depth"],
                "learning_rate": config["learning_rate"]
            })

            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            })

            signature = infer_signature(
                X_train,
                model.predict(X_train)
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.head(5),
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            print(
                f"{config['run_name']} | "
                f"RMSE: {rmse:.4f} | "
                f"MAE: {mae:.4f} | "
                f"R2: {r2:.4f}"
            )

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    results = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"]
    )

    print("\nBest Run based on lowest RMSE:")

    if not results.empty:
        print(
            results[
                [
                    "tags.mlflow.runName",
                    "params.max_depth",
                    "params.learning_rate",
                    "metrics.rmse",
                    "metrics.mae",
                    "metrics.r2"
                ]
            ].head(1).to_string(index=False)
        )


if __name__ == "__main__":
    main()
