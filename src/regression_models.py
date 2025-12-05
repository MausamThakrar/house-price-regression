from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


@dataclass
class RegressionResult:
    name: str
    mae: float
    rmse: float
    r2: float


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the housing dataset from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path.resolve()}")
    df = pd.read_csv(path)
    return df


def train_test_split_xy(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Split features and target into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


def build_linear_regression_model() -> Tuple[StandardScaler, LinearRegression]:
    """Return a scaler and a linear regression model."""
    scaler = StandardScaler()
    model = LinearRegression()
    return scaler, model


def build_random_forest_model(random_state: int = 42) -> RandomForestRegressor:
    """Return a random forest regressor."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
    )
    return model


def evaluate_regression_model(
    y_true: np.ndarray, y_pred: np.ndarray, name: str
) -> RegressionResult:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return RegressionResult(name=name, mae=mae, rmse=rmse, r2=r2)


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred_lr: np.ndarray, y_pred_rf: np.ndarray) -> None:
    """
    Scatter plot of predicted vs actual prices for two models.
    """
    plt.figure()
    plt.scatter(y_true, y_pred_lr, label="Linear Regression", alpha=0.8)
    plt.scatter(y_true, y_pred_rf, label="Random Forest", alpha=0.8)
    max_val = max(y_true.max(), y_pred_lr.max(), y_pred_rf.max())
    min_val = min(y_true.min(), y_pred_lr.min(), y_pred_rf.min())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual Price (EUR)")
    plt.ylabel("Predicted Price (EUR)")
    plt.title("Predicted vs Actual House Prices")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importances(
    model: RandomForestRegressor, feature_names: pd.Index
) -> None:
    """Bar plot for random forest feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]

    plt.figure()
    plt.bar(range(len(sorted_features)), sorted_importances)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()


def main():
    # 1. Load data
    df = load_data("data/house_prices.csv")
    print("First 5 rows of the dataset:")
    print(df.head())

    # 2. Train/test split
    X_train, X_test, y_train, y_test, feature_names = train_test_split_xy(
        df, target_column="price_eur", test_size=0.25, random_state=42
    )

    # 3. Linear Regression (with scaling)
    scaler_lr, lr_model = build_linear_regression_model()
    X_train_scaled = scaler_lr.fit_transform(X_train)
    X_test_scaled = scaler_lr.transform(X_test)

    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_result = evaluate_regression_model(y_test, y_pred_lr, name="Linear Regression")

    # 4. Random Forest Regression
    rf_model = build_random_forest_model(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_result = evaluate_regression_model(y_test, y_pred_rf, name="Random Forest")

    # 5. Print evaluation results
    print("\n=== MODEL EVALUATION ===")
    for res in (lr_result, rf_result):
        print(
            f"{res.name}: "
            f"MAE={res.mae:.2f}, RMSE={res.rmse:.2f}, R^2={res.r2:.3f}"
        )

    # 6. Visualisations
    plot_predicted_vs_actual(y_test, y_pred_lr, y_pred_rf)
    plot_feature_importances(rf_model, feature_names)


if __name__ == "__main__":
    main()
