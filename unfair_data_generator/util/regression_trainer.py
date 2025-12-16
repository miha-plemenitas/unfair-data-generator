import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from .helpers import get_group_name
except ImportError:
    from unfair_data_generator.util.helpers import get_group_name


def train_and_evaluate_regression(X, y, Z):
    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(
        X, y, Z, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {}
    all_groups = np.unique(Z)

    for g in np.unique(Z_test):
        mask = Z_test == g
        group_name = get_group_name(all_groups, g)

        mse = mean_squared_error(y_test[mask], y_pred[mask])
        mae = mean_absolute_error(y_test[mask], y_pred[mask])
        bias = np.mean(y_pred[mask] - y_test[mask])

        metrics[group_name] = {
            "MSE": mse,
            "MAE": mae,
            "Bias": bias,
            "Samples": mask.sum()
        }

    return metrics
