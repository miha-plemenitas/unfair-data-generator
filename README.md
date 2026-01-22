<h1 align="center">
    Unfair Data Generator
</h1>

<p align="center">
    <img alt="PyPI version" src="https://img.shields.io/pypi/v/unfair-data-generator.svg" />
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/unfair-data-generator.svg">
    <a href="https://pepy.tech/project/unfair-data-generator">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/unfair-data-generator.svg">
    </a>
    <img alt="Downloads" src="https://static.pepy.tech/badge/unfair-data-generator">
    <img alt="unfair-data-generator" src="https://github.com/tiazv/unfair-data-generator/actions/workflows/test.yml/badge.svg" />
    <img alt="Documentation status" src="https://readthedocs.org/projects/unfair-data-generator/badge/?version=latest" />
</p>

<p align="center">
    <img alt="Repository size" src="https://img.shields.io/github/repo-size/tiazv/unfair-data-generator" />
    <img alt="License" src="https://img.shields.io/github/license/tiazv/unfair-data-generator.svg" />
    <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/w/tiazv/unfair-data-generator.svg">
    <a href="http://isitmaintained.com/project/tiazv/unfair-data-generator">
        <img alt="Percentage of issues still open" src="http://isitmaintained.com/badge/open/tiazv/unfair-data-generator.svg">
    </a>
    <a href="http://isitmaintained.com/project/tiazv/unfair-data-generator">
        <img alt="Average time to resolve an issue" src="http://isitmaintained.com/badge/resolution/tiazv/unfair-data-generator.svg">
    </a>
    <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/tiazv/unfair-data-generator.svg"/>
</p>

<p align="center">
    <a href="#-about"> About</a> •
    <a href="#-installation"> Installation</a> •
    <a href="#-usage"> Usage</a> •
    <a href="#-supported-equality-types">⚖️ Supported Equality Types</a> •
    <a href="#-community-guidelines"> Community Guidelines</a> •
    <a href="#-license"> License</a>
</p>

##  About
**Unfair Data Generator** is a Python library designed for generating biased **classification and regression** datasets with intentional unfairness patterns. This tool extends scikit-learn's `make_classification` function to include sensitive group information and fairness constraints, and expands the same idea to regression by generating continuous targets with controlled group-dependent bias and noise. ⚖️

In short: classification uses discrete labels with group-level outcome disparities, while regression uses continuous targets with group-level error or bias differences. This makes the library useful for both fairness research and education across problem types.

* **Free software:** MIT license
* **Documentation:** [https://unfair-data-generator.readthedocs.io](https://unfair-data-generator.readthedocs.io)
* **Python**: 3.11, 3.12
* **Dependencies**: listed in [CONTRIBUTING.md](./CONTRIBUTING.md#dependencies)
* **Operating systems**: Windows, Ubuntu, macOS

## ✨ Features
- **Biased Dataset Generation**: Create biased datasets for both classification and regression across sensitive groups. 🗃️
- **Fairness Evaluation**: Built-in tools for evaluating model fairness across different groups in classifiers and regressors. ⚖️
- **Visualization**: Visualization capabilities for understanding bias patterns and fairness metrics. 📈
- **Flexible Configuration**: Support for various equality types in classification and regression (demographic parity, equal opportunity, equalized odds; group bias, heteroscedastic noise). ⚙️
- **Leaky Features**: Generate features that leak sensitive information to simulate real-world bias. 🔓
- **Multiple Groups**: Support for 2-5 sensitive groups with intuitive weather-based naming. 🌦️
- **Scikit-learn Compatible**: Extends familiar scikit-learn patterns and interfaces. 🎯
- **Regression Base Functions**: Linear, logistic/sigmoid, and exponential target generation.
- **Regression Metrics**: Group-wise evaluation using MAE, RMSE, mean residual, and R2.

##  Installation
### pip
To install `unfair-data-generator` using pip, run the following command:
```bash
pip install unfair-data-generator
```

##  Usage
The following examples demonstrate how to generate biased datasets and evaluate fairness using `unfair-data-generator`. More examples can be found in the [examples](./examples) directory.

### Unfair Classification Example
```python
from unfair_data_generator.unfair_classification import make_unfair_classification
from unfair_data_generator.util.helpers import get_params_for_certain_equality_type
from unfair_data_generator.util.model_trainer import train_and_evaluate_model_with_classifier
from unfair_data_generator.util.visualizer import (
    visualize_TPR_FPR_metrics, 
    visualize_accuracy, 
    visualize_groups_separately,
    visualize_group_classes
)

# Configure dataset parameters
fairness_type = "Demographic parity"
n_sensitive_groups = 3

# Generate group-specific parameters for fairness violation
group_params = get_params_for_certain_equality_type(fairness_type, n_sensitive_groups)

# Generate biased dataset
X, y, Z, centroids = make_unfair_classification(
    n_samples=5000,
    n_features=10,
    n_informative=3,
    n_leaky=2,
    random_state=42,
    group_params=group_params,
    return_sensitive_group_centroids=True,
)

# Visualize group-specific patterns
visualize_groups_separately(X, y, Z)
visualize_group_classes(X, y, Z, centroids)

# Train model and evaluate fairness
metrics = train_and_evaluate_model_with_classifier(X, y, Z)

# Visualize fairness metrics
title = f"{fairness_type} with {n_sensitive_groups} sensitive groups"
visualize_TPR_FPR_metrics(metrics, title)
visualize_accuracy(metrics, title)
```

### Unfair Regression Example
```python
from unfair_data_generator.unfair_regression import make_unfair_regression
from unfair_data_generator.util.model_trainer import train_and_evaluate_model_with_regressor
from unfair_data_generator.util.visualizer import visualize_regression_metrics

X, y, Z = make_unfair_regression(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    fairness_type="Group bias",
    base_function="linear",
    n_sensitive_groups=3,
    random_state=42
)

metrics = train_and_evaluate_model_with_regressor(X, y, Z)

# Visualize regression fairness metrics
title = f"Regression fairness: {n_sensitive_groups} groups"
visualize_regression_metrics(metrics, title)

print(metrics)
```

## ⚖️ Supported Equality Types
The library supports generating classification and regression datasets that systematically violate specific fairness criteria. Each type creates different bias patterns:

### Classification
- **Equal quality**   
  Different classification performance across groups.
- **Demographic parity**  
  Unequal positive prediction rates across groups.
- **Equal opportunity**  
  Unequal true positive rates across groups.
- **Equalized odds**  
  Unequal true positive and false positive rates across groups.

### Regression
Each type creates different group-level bias or error patterns:
- **Equal MSE**  
  Same noise scale and zero bias across groups.
- **Group bias**  
  Different constant bias per group with equal noise scale.
- **Heteroscedastic noise**  
  Different noise scales per group with zero bias.

## 🫂 Community Guidelines
### Contributing
To contribure to the software, please read the [contributing guidelines](./CONTRIBUTING.md).

### Reporting Issues
If you encounter any issues with the library, please report them using the [issue tracker](https://github.com/tiazv/unfair-data-generator/issues). Include a detailed description of the problem, including the steps to reproduce the problem, the stack trace, and details about your operating system and software version.

### Seeking Support
If you need support, please first refer to the [documentation](https://unfair-data-generator.readthedocs.io). If you still require assistance, please open an issue on the [issue tracker](https://github.com/tiazv/unfair-data-generator/issues) with the `question` tag. For private inquiries, you can contact us via e-mail at [saso.karakatic@um.si](mailto:saso.karakatic@um.si) or [tadej.lahovnik1@um.si](mailto:tadej.lahovnik1@um.si).

## 📜 License
This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!
