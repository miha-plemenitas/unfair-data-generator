import numpy as np

try:
    from .unfair_classification import make_unfair_classification
    from .util.helpers import get_group_name, get_params_for_certain_regression_type
except ImportError:
    from unfair_data_generator.unfair_classification import make_unfair_classification
    from unfair_data_generator.util.helpers import (
        get_group_name,
        get_params_for_certain_regression_type,
    )


def _compute_base_signal(X, n_informative, rng, base_function):
    informative_features = X[:, :n_informative]
    coefficients = rng.normal(size=n_informative)
    linear_response = informative_features @ coefficients

    if base_function == "linear":
        return linear_response
    if base_function in {"logistic", "sigmoid"}:
        return 1.0 / (1.0 + np.exp(-linear_response))
    if base_function == "exponential":
        return np.exp(linear_response)

    raise ValueError(
        f"Unsupported base_function: {base_function}. "
        "Choose from {'linear', 'logistic', 'sigmoid', 'exponential'}."
    )


def make_unfair_regression(
    n_samples=100,
    n_features=20,
    *,
    n_informative=5,
    n_redundant=2,
    n_repeated=0,
    n_leaky=0,
    noise=0.1,
    random_state=None,
    fairness_type="Equal MSE",
    base_function="linear",
    group_params=None,
    n_sensitive_groups=2,
    return_sensitive_group_centroids=False,
    append_sensitive_to_X=False,
    shuffle=True,
):
    """
    Generate an unfair regression dataset with sensitive groups.

    This reuses the unfair classification generator for feature and group
    structure, then generates a continuous target with group-dependent bias
    and noise. Regression unfairness is controlled via ``fairness_type`` or
    explicit ``group_params`` (bias and noise scale per group).

    Parameters
    ----------
    fairness_type : {"Equal MSE", "Group bias", "Heteroscedastic noise"}
        Determines how group bias and noise are generated when group_params is None.
    group_params : dict, optional
        Mapping of group name -> {"bias": float, "noise_scale": float}.
    base_function : {"linear", "logistic", "sigmoid", "exponential"}
        Functional form used for the base regression signal.

    Returns
    -------
    X : ndarray
    y : ndarray (continuous)
    Z : ndarray (sensitive groups) when append_sensitive_to_X=False
    centroids : optional
    """
    rng = np.random.default_rng(random_state)

    if return_sensitive_group_centroids:
        X, _, Z, centroids = make_unfair_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_leaky=n_leaky,
            random_state=random_state,
            n_sensitive_groups=n_sensitive_groups,
            fairness_type="Equal quality",
            group_params=None,
            append_sensitive_to_X=False,
            return_sensitive_group_centroids=True,
            shuffle=shuffle,
        )
    else:
        X, _, Z = make_unfair_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_leaky=n_leaky,
            random_state=random_state,
            n_sensitive_groups=n_sensitive_groups,
            fairness_type="Equal quality",
            group_params=None,
            append_sensitive_to_X=False,
            return_sensitive_group_centroids=False,
            shuffle=shuffle,
        )
        centroids = None

    if group_params is None:
        group_params = get_params_for_certain_regression_type(
            fairness_type, n_sensitive_groups
        )

    base_signal = _compute_base_signal(
        X=X,
        n_informative=n_informative,
        rng=rng,
        base_function=base_function,
    )

    y = np.zeros(n_samples)
    unique_groups = np.unique(Z)

    for group in unique_groups:
        mask = Z == group
        group_name = get_group_name(unique_groups, group)

        if group_name not in group_params:
            raise ValueError(
                f"Missing regression params for group '{group_name}'."
            )

        params = group_params[group_name]
        group_bias = float(params.get("bias", 0.0))
        noise_scale = float(params.get("noise_scale", 1.0))
        group_noise = noise * noise_scale

        y[mask] = (
            base_signal[mask]
            + group_bias
            + rng.normal(0, group_noise, size=mask.sum())
        )

    if append_sensitive_to_X:
        X = np.column_stack((X, Z))

    if return_sensitive_group_centroids:
        if append_sensitive_to_X:
            return X, y, centroids
        return X, y, Z, centroids

    if append_sensitive_to_X:
        return X, y

    return X, y, Z
