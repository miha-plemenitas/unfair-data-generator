import numpy as np
import numpy.testing as npt

from unfair_data_generator.unfair_regression import make_unfair_regression
from unfair_data_generator.util.model_trainer import train_and_evaluate_model_with_regressor


SEED = 42
N_SAMPLES = 500
N_FEATURES = 10
N_GROUPS = 3
N_INFORMATIVE = 3


def _generate_dataset(append_sensitive, return_centroids):
    return make_unfair_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=2,
        n_leaky=0,
        random_state=SEED,
        n_sensitive_groups=N_GROUPS,
        fairness_type="Group bias",
        base_function="linear",
        append_sensitive_to_X=append_sensitive,
        return_sensitive_group_centroids=return_centroids,
    )


def test_returns_X_y_Z():
    X, y, Z = _generate_dataset(
        append_sensitive=False, return_centroids=False
    )

    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert y.shape == (N_SAMPLES,)
    assert Z.shape == (N_SAMPLES,)

    assert np.min(Z) >= 0 and np.max(Z) < N_GROUPS
    assert np.issubdtype(y.dtype, np.floating)
    assert np.issubdtype(Z.dtype, np.integer)


def test_returns_X_y():
    X, y, Z = _generate_dataset(
        append_sensitive=False, return_centroids=False
    )
    X_with_sensitive, y_with_sensitive = _generate_dataset(
        append_sensitive=True, return_centroids=False
    )

    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert X_with_sensitive.shape == (N_SAMPLES, N_FEATURES + 1)
    assert y.shape == (N_SAMPLES,) and y_with_sensitive.shape == (N_SAMPLES,)

    npt.assert_array_equal(y, y_with_sensitive)
    npt.assert_array_equal(X_with_sensitive[:, :-1], X)
    npt.assert_array_equal(X_with_sensitive[:, -1].astype(int), Z)
    assert np.min(X_with_sensitive[:, -1]) >= 0 and np.max(X_with_sensitive[:, -1]) < N_GROUPS


def test_returns_X_y_Z_centroids():
    X, y, Z, centroids = _generate_dataset(
        append_sensitive=False, return_centroids=True
    )

    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert y.shape == (N_SAMPLES,)
    assert Z.shape == (N_SAMPLES,)
    assert isinstance(centroids, dict) and len(centroids) == N_GROUPS

    for group, centroid_array in centroids.items():
        assert centroid_array.shape[1] == N_INFORMATIVE


def test_returns_X_y_centroids():
    X, y, centroids = _generate_dataset(
        append_sensitive=True, return_centroids=True
    )

    assert X.shape == (N_SAMPLES, N_FEATURES + 1)
    assert y.shape == (N_SAMPLES,)
    assert isinstance(centroids, dict) and len(centroids) == N_GROUPS

    for group, centroid_array in centroids.items():
        assert centroid_array.shape[1] == N_INFORMATIVE


def test_regression_metrics_keys():
    X, y, Z = _generate_dataset(
        append_sensitive=False, return_centroids=False
    )
    metrics = train_and_evaluate_model_with_regressor(X, y, Z)

    assert metrics
    for group_name, group_metrics in metrics.items():
        assert set(group_metrics.keys()) == {
            "MAE",
            "RMSE",
            "Mean residual",
            "R2",
            "Samples",
        }
        assert isinstance(group_metrics["Samples"], int)
        assert group_metrics["Samples"] >= 0
