# skmiscpy

[![PyPI](https://img.shields.io/pypi/v/skmiscpy.svg)](https://pypi.org/project/skmiscpy/) ![Python Versions](https://img.shields.io/pypi/pyversions/skmiscpy) ![License](https://img.shields.io/pypi/l/skmiscpy) [![Build](https://github.com/shafayetShafee/skmiscpy/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shafayetShafee/skmiscpy/actions/workflows/ci-cd.yml) [![codecov](https://codecov.io/github/shafayetShafee/skmiscpy/graph/badge.svg?token=OAZ6C1KHC9)](https://codecov.io/github/shafayetShafee/skmiscpy)

Contains a few functions useful for data-analysis, causal inference etc.

## Installation

```bash
pip install skmiscpy
```

## Usage

So far, `skmiscpy` can be used to do a basic causal analysis. Here very simple examples are shown for demonstration purposes.
Check [Causal Analysis Workflow & Estimating ATE Using `skmiscpy`](https://skmiscpy.readthedocs.io/en/latest/example.html) for
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

```python
sample_df = pd.DataFrame({
    'age': np.random.randint(18, 66, size=100),
    'weight': np.round(np.random.uniform(120, 200, size=100), 1),
    'gender': np.random.choice(['male', 'female'], size=100),
    'race': np.random.choice(
        ['white', 'black', 'hispanic'],
        size=100, p=[0.4, 0.3, 0.3]
    ),
    'educ_level': np.random.choice(
        ['bachelor', 'master', 'doctorate'],
        size=100, p=[0.3, 0.4, 0.3]
    ),
    'ps_wts': np.round(np.random.uniform(0.1, 1.0, size=100), 2),
    'group': np.random.choice(['treated', 'control'], size=100),
    'date': pd.date_range(start='2024-01-01', periods=100, freq='D')
})

# 1. Basic usage with unadjusted SMD only:
compute_smd(sample_df, vars=['age', 'weight', 'gender'], group='group', estimand='ATE')

# 2. Including weights for adjusted SMD:
compute_smd(
    sample_df, 
    vars=['age', 'weight', 'gender'], 
    group='group', wt_var='ps_wts',
    estimand='ATE'
)

# 3. Including categorical variables for adjusted SMD:
compute_smd(
    sample_df,
    vars=['age', 'weight', 'gender'],
    group='group',
    wt_var='ps_wts',
    cat_vars=['race', 'educ_level'],
    estimand='ATE'
)
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
