# skmiscpy

Contains a few functions useful for data-analysis, causal inference etc.

## Installation

```bash
pip install skmiscpy
```

## Usage

So far, `skmiscpy` can be used to do a basic causal analysis. Here very simple examples are shown for demonstration purposes.
Check (Causal Analysis Workflow & Estimating ATE Using `skmiscpy`)[https://skmiscpy.readthedocs.io/en/latest/example.html] for
better understanding.

``` python
import pandas as pd
from skmiscpy import compute_smd, plot_smd
from skmiscpy import plot_mirror_histogram
```

### Draw a mirror histogram

``` python
data = pd.DataFrame({
    'treatment': [1, 1, 0, 0, 1, 0],
    'propensity_score': [2.0, 3.5, 3.0, 2.2, 2.2, 3.3]
})

plot_mirror_histogram(data=data, var='propensity_score', group='treatment')

# Draw a weighted mirror histogram
data_with_weights = pd.DataFrame({
    'treatment': [1, 1, 0, 0, 1, 0],
    'propensity_score': [2.0, 3.5, 3.0, 2.2, 2.2, 3.3],
    'weights': [1.0, 1.5, 2.0, 1.2, 1.1, 0.8]
})

plot_mirror_histogram(
    data=data_with_weights, var='propensity_score', group='treatment', weights='weights',
    xlabel='Propensity Score', ylabel='Weighted Count', title='Weighted Mirror Histogram'
)
```
### Compute Standardized Mean Difference (SMD)

``` python
data = pd.DataFrame({
    'group': [1, 0, 1, 0, 1, 0],
    'age': [23, 35, 45, 50, 22, 30],
    'bmi': [22.5, 27.8, 26.1, 28.5, 24.3, 29.0],
    'blood_pressure': [120, 130, 140, 135, 125, 133],
    'weights': [1.2, 0.8, 1.5, 0.7, 1.0, 0.9]
})

# Compute SMD for 'age', 'bmi', and 'blood_pressure' under ATE estimand
smd_results = compute_smd(data, vars=['age', 'bmi', 'blood_pressure'], group='group', estimand='ATE')

# Compute SMD adjusted by weights
smd_results_with_weights = compute_smd(data, vars=['age', 'bmi', 'blood_pressure'], group='group', wt_var='weights')

print(smd_results)
print(smd_results_with_weights)
```

### Create a love plot (point plot of SMD)

``` python
data = pd.DataFrame({
    'variables': ['age', 'bmi', 'blood_pressure'],
    'unadjusted_smd': [0.25, 0.4, 0.1],
    'adjusted_smd': [0.05, 0.2, 0.08]
})

plot_smd(data)

## Adding a reference line at 0.1
plot_smd(data, add_ref_line=True, ref_line_value=0.1)

## Customizing the Seaborn plot with additional keyword arguments
plot_smd(data, add_ref_line=True, ref_line_value=0.1, palette='coolwarm', markers=['o', 's'])

```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`skmiscpy` was created by Shafayet Khan Shafee. It is licensed under the terms of the MIT license.

## Credits

`skmiscpy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
