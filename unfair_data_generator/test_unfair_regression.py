from unfair_data_generator.unfair_regression import make_unfair_regression
from unfair_data_generator.util.regression_trainer import train_and_evaluate_regression

# Generate unfair regression data
X, y, Z = make_unfair_regression(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    fairness_type="Group bias",
    n_sensitive_groups=3,
    random_state=42
)

# Basic sanity checks
print("Shapes:")
print("X:", X.shape)
print("y:", y.shape)
print("Z:", Z.shape)

# Train regression model and evaluate fairness
metrics = train_and_evaluate_regression(X, y, Z)

print("\nRegression fairness metrics:")
for group, vals in metrics.items():
    print(f"\nGroup: {group}")
    for k, v in vals.items():
        print(f"  {k}: {v}")
