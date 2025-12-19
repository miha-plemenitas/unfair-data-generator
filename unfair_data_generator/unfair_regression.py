import numpy as np

try:
    from .unfair_classification import make_unfair_classification
except ImportError:
    from unfair_data_generator.unfair_classification import make_unfair_classification


def make_unfair_regression(
    n_samples=100,
    n_features=20,
    *,
    n_informative=5,
    noise=0.1,
    random_state=None,
    fairness_type="Equal MSE",
    base_function="linear",
    group_params=None,
    n_sensitive_groups=2,
    return_sensitive_group_centroids=False
):
    """
    Generate an unfair regression dataset by reusing the unfair classification
    generator for X and Z, and generating a group-dependent regression target y.

    Parameters
    ----------
    base_function : {"linear", "logistic", "exponential"}
        Functional form used to generate the base regression signal shared
        across all sensitive groups.

    Returns
    -------
    X : ndarray
    y : ndarray (continuous)
    Z : ndarray (sensitive groups)
    centroids : optional
    """

    rng = np.random.default_rng(random_state)

    # 1. Generate X and Z using classification generator
    X, _, Z, centroids = make_unfair_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
        n_sensitive_groups=n_sensitive_groups,
        group_params=group_params,
        append_sensitive_to_X=False,
        return_sensitive_group_centroids=True
    )

    # 2. Generate base regression signal (shared across groups)
    w = rng.normal(size=n_informative)
    linear_signal = X[:, :n_informative] @ w

    if base_function == "linear":
        base_signal = linear_signal

    elif base_function == "logistic":
        base_signal = 1.0 / (1.0 + np.exp(-linear_signal))

    elif base_function == "exponential":
        base_signal = np.exp(linear_signal)

    else:
        raise ValueError(
            f"Unsupported base_function: {base_function}. "
            "Choose from {'linear', 'logistic', 'exponential'}."
        )

    # 3. Group-specific bias and noise
    y = np.zeros(n_samples)
    unique_groups = np.unique(Z)

    for g in unique_groups:
        mask = Z == g

        if fairness_type == "Equal MSE":
            group_noise = noise
            group_bias = 0.0

        elif fairness_type == "Group bias":
            group_noise = noise
            group_bias = g * 0.5

        elif fairness_type == "Heteroscedastic noise":
            group_noise = noise * (1 + g)
            group_bias = 0.0

        else:
            raise ValueError(f"Unsupported fairness_type: {fairness_type}")

        y[mask] = (
            base_signal[mask]
            + group_bias
            + rng.normal(0, group_noise, size=mask.sum())
        )

    if return_sensitive_group_centroids:
        return X, y, Z, centroids

    return X, y, Z
