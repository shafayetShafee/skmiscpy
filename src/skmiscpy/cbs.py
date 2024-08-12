import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Union
from skmiscpy.utils import _check_param_type, _check_required_columns
from skmiscpy.utils import _check_variance_positive, _check_proportion_within_range


def compute_smd(
    data: pd.DataFrame,
    vars: List[str],
    group: str,
    wt_var: str = None,
    estimand: str = "ATE",
) -> pd.DataFrame:
    """
    Computes the standardized mean difference (SMD) for a list of variables.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the columns specified in `vars`, `group`, and optionally `wt_var`.

    vars : List[str]
        A list of strings representing the variable names for which to calculate the SMD.

    group : str
        The name of the binary group column (e.g., treatment vs. control).

    wt_var : str, optional
        The name of the column containing weights. Defaults to None.

    estimand : str, optional
        The estimand type. Currently, only 'ATE' (Average Treatment Effect) is supported. Defaults to 'ATE'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'variable': The name of the variable.
        - 'unadjusted_smd': The standardized mean difference without adjustment.
        - 'adjusted_smd': The standardized mean difference with adjustment (if `wt_var` is provided).

    Examples
    --------
    1. Basic usage with unadjusted SMD only:

    >>> import pandas as pd
    >>> from skmiscpy import plot_smd

    >>> data = pd.DataFrame({
    >>>     'variable1': [1, 2, 3, 4],
    >>>     'variable2': [2, 3, 4, 5],
    >>>     'group': [0, 1, 0, 1]
    >>> })

    >>> compute_smd(data, vars=['variable1', 'variable2'], group='group')
    # Returns a DataFrame with unadjusted SMD values for 'variable1' and 'variable2'.

    2. Including weights for adjusted SMD:

    >>> data = pd.DataFrame({
    >>>     'variable1': [1, 2, 3, 4],
    >>>     'variable2': [2, 3, 4, 5],
    >>>     'group': [0, 1, 0, 1],
    >>>     'weights': [1.5, 2.0, 1.2, 1.8]
    >>> })

    >>> compute_smd(data, vars=['variable1', 'variable2'], group='group', wt_var='weights')
    # Returns a DataFrame with both unadjusted and adjusted SMD values for 'variable1' and 'variable2'.

    3. Single variable input:

    >>> data = pd.DataFrame({
    >>>     'variable1': [1, 2, 3, 4],
    >>>     'group': [0, 1, 0, 1]
    >>> })

    >>> compute_smd(data, vars='variable1', group='group')
    # Returns a DataFrame with unadjusted SMD values for 'variable1'.
    """
    data = _check_smd_data(
        data=data, group=group, vars=vars, wt_var=wt_var, estimand=estimand
    )

    smd_results = []
    vars = [vars] if isinstance(vars, str) else vars

    for var in vars:
        if wt_var is not None:
            unadjusted_smd = _calc_smd_covar(
                data=data, group=group, covar=var, estimand=estimand
            )
            adjusted_smd = _calc_smd_covar(
                data=data, group=group, covar=var, wt_var=wt_var, estimand=estimand
            )
            smd_results.append(
                {
                    "variables": var,
                    "unadjusted_smd": unadjusted_smd,
                    "adjusted_smd": adjusted_smd,
                }
            )
        else:
            unadjusted_smd = _calc_smd_covar(
                data=data, group=group, covar=var, estimand=estimand
            )
            smd_results.append({"variables": var, "unadjusted_smd": unadjusted_smd})

    smd_df = pd.DataFrame(smd_results)

    return smd_df


def _calc_smd_covar(
    data: pd.DataFrame,
    group: str,
    covar: str,
    wt_var: str = None,
    estimand: str = "ATE",
) -> float:
    """
    Calculate the Standardized Mean Difference (SMD) for a covariate between two groups.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the group, covariate, and optional weights.
    group : str
        The column name indicating the group variable. This column must be binary.
    covar : str
        The column name of the covariate for which the SMD is calculated.
    wt_var : str, optional
        The column name of the weights. If None, only the unadjusted SMD is calculated. Defaults to None.
    estimand : str, optional
        The causal estimand to use. Supports only "ATE" (Average Treatment Effect) currently. Defaults to "ATE".

    Returns
    -------
    float
        The calculated Standardized Mean Difference.

    Raises
    ------
    TypeError:
        If `data` params is not a pd.DataFrame, if `group`, `covar`, or `wt_var` params are not strings.
    ValueError
        if `estimand` is invalid, if the `group` column does not contain binary values, or if the weight column
        has non-positive or zero values.
    ValueError
        If `s2_1` or `s2_0` is less than or equal to zero.
    """

    grp_1_dt = data[data[group] == 1]
    grp_0_dt = data[data[group] == 0]
    covar_1 = grp_1_dt[covar]
    covar_0 = grp_0_dt[covar]

    m1 = covar_1.mean()
    m0 = covar_0.mean()
    s2_1 = covar_1.var()
    s2_0 = covar_0.var()

    custom_msg_1 = f"{covar} for group 1"
    custom_msg_0 = f"{covar} for group 0"
    bin_custom_msg_1 = f"proportion of {covar} for group 1"
    bin_custom_msg_0 = f"proportion of {covar} for group 0"
    wt_bin_custom_msg_1 = f"weighted proportion of {covar} for group 1"
    wt_bin_custom_msg_0 = f"weighted proportion of {covar} for group 0"

    if data[covar].dropna().nunique() == 2:
        _check_proportion_within_range(m1, bin_custom_msg_1)
        _check_proportion_within_range(m0, bin_custom_msg_0)
    else:
        _check_variance_positive(s2_1, custom_msg_1)
        _check_variance_positive(s2_0, custom_msg_0)

    if wt_var is not None:
        wt_m1 = np.average(covar_1, weights=grp_1_dt[wt_var])
        wt_m0 = np.average(covar_0, weights=grp_0_dt[wt_var])
        if data[covar].dropna().nunique() == 2:
            _check_proportion_within_range(wt_m1, wt_bin_custom_msg_1)
            _check_proportion_within_range(wt_m0, wt_bin_custom_msg_0)
            return _calc_smd_bin_covar(estimand, m1=m1, m0=m0, wt_m1=wt_m1, wt_m0=wt_m0)
        else:
            return _calc_smd_cont_covar(
                estimand, m1=wt_m1, m0=wt_m0, s2_1=s2_1, s2_0=s2_0
            )
    else:
        if data[covar].dropna().nunique() == 2:
            return _calc_smd_bin_covar(estimand, m1=m1, m0=m0)
        else:
            return _calc_smd_cont_covar(estimand, m1=m1, m0=m0, s2_1=s2_1, s2_0=s2_0)


def _check_smd_data(
    data: pd.DataFrame,
    group: str,
    vars: Union[str, List[str]],
    wt_var: str = None,
    estimand: str = "ATE",
) -> pd.DataFrame:
    """
    Validate the input data for SMD calculation.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    group : str
        The column name for the group variable, which must be binary.
    vars : str or list of str
        The column name(s) of the covariates to be checked. It can be a single string or a list of strings.
    wt_var : str, optional
        The column name for weights, if applicable. Default is None.
    estimand : str, optional
        The causal estimand to use. Default is "ATE". Currently, only "ATE" is supported.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame with adjusted group values if necessary.

    Raises
    ------
    TypeError:
        If `data` params is not a pd.DataFrame, if `group`, `covar`, or `wt_var` params are not strings.
    ValueError
        if `estimand` is invalid, if the `group` column does not contain binary values, or if the weight column
        has non-positive or zero values.
    """

    _check_param_type({"data": data}, pd.DataFrame)
    _check_param_type({"group": group}, str)

    vars = [vars] if isinstance(vars, str) else vars

    if not all(isinstance(v, str) for v in vars):
        raise TypeError("`vars` must be a string or a list of strings")

    required_columns = set(vars + [group])
    if wt_var is not None:
        required_columns.add(wt_var)

    _check_required_columns(data, list(required_columns))

    for col in [group] + vars + ([wt_var] if wt_var else []):
        if data[col].isnull().any():
            raise ValueError(f"The '{col}' column contains missing values.")

    unique_groups = data[group].dropna().unique()
    if len(unique_groups) == 2:
        if set(unique_groups) != {0, 1}:
            min_val, max_val = min(unique_groups), max(unique_groups)
            data[group] = data[group].apply(lambda x: 0 if x == min_val else 1)
    else:
        raise ValueError(
            f"The '{group}' column must be a binary column for valid SMD calculation."
        )

    for var in vars:
        unique_vals = data[var].dropna().unique()
        if len(unique_vals) == 2:
            min_val = min(unique_vals)
            data[var] = data[var].apply(lambda x: 0 if x == min_val else 1)

        if not pd.api.types.is_numeric_dtype(data[var]):
            if len(unique_vals) == 2:
                raise ValueError(
                    f"The '{var}' column could not be converted into 0-1 binary column."
                )
            else:
                raise ValueError(f"The '{var}' column must be numeric.")

    if wt_var is not None:
        if not pd.api.types.is_numeric_dtype(data[wt_var]):
            raise ValueError(f"The '{wt_var}' column must be numeric.")

        if (data[wt_var] <= 0).any():
            raise ValueError(
                f"The '{wt_var}' column contains negative weight values. The weight values must be positive"
            )

    return data


def _calc_smd_bin_covar(estimand, *args, **kwargs):
    estimand_to_function = {
        "ATE": _calc_smd_bin_covar_ate,
        "ATT": _calc_smd_bin_covar_att,
        "ATC": _calc_smd_bin_covar_atc,
    }
    function_to_call = estimand_to_function.get(estimand)

    if function_to_call is not None:
        return function_to_call(*args, **kwargs)
    else:
        raise ValueError(f"Invalid estimand value: {estimand}")


def _calc_smd_cont_covar(estimand, *args, **kwargs):
    estimand_to_function = {
        "ATE": _calc_smd_cont_covar_ate,
        "ATT": _calc_smd_cont_covar_att,
        "ATC": _calc_smd_cont_covar_atc,
    }
    function_to_call = estimand_to_function.get(estimand)

    if function_to_call is not None:
        return function_to_call(*args, **kwargs)
    else:
        raise ValueError(f"Invalid estimand value: {estimand}")


def _calc_smd_bin_covar_ate(
    m1: float, m0: float, wt_m1: float = None, wt_m0: float = None
) -> float:
    """
    Calculate the Standardized Mean Difference (SMD) for binary covariates using the Average Treatment Effect (ATE).

    Parameters
    ----------
    m1 : float
        The mean of the covariate for the treatment group. Must be between 0 and 1.
    m0 : float
        The mean of the covariate for the control group. Must be between 0 and 1.
    wt_m1 : float, optional
        The weighted mean of the covariate for the treatment group. If not provided, `m1` is used. Must be between 0 and 1.
    wt_m0 : float, optional
        The weighted mean of the covariate for the control group. If not provided, `m0` is used. Must be between 0 and 1.

    Returns
    -------
    float
        The Standardized Mean Difference (SMD).
    """
    wt_m1 = m1 if wt_m1 is None else wt_m1
    wt_m0 = m0 if wt_m0 is None else wt_m0

    pooled_var = (m1 * (1 - m1)) + (m0 * (1 - m0))
    std_factor = np.sqrt(pooled_var / 2)

    smd = _calc_raw_smd(a=wt_m1, b=wt_m0, std_factor=std_factor)
    return smd


def _calc_smd_cont_covar_ate(m1: float, m0: float, s2_1: float, s2_0: float) -> float:
    """
    Calculate the standardized mean difference (SMD) for continuous covariates
    with an Average Treatment Effect (ATE) estimand.

    Parameters
    ----------
    m1 : float
        The mean of the covariate for group 1.
    m0 : float
        The mean of the covariate for group 0.
    s2_1 : float
        The variance of the covariate for group 1. Must be strictly positive.
    s2_0 : float
        The variance of the covariate for group 0. Must be strictly positive.

    Returns
    -------
    float
        The standardized mean difference (SMD).
    """
    std_factor = np.sqrt((s2_1 + s2_0) / 2)
    smd = _calc_raw_smd(a=m1, b=m0, std_factor=std_factor)
    return smd


def _calc_smd_bin_covar_att(*args, **kwargs):
    raise NotImplementedError("SMD for ATT estimand is not yet implemented.")


def _calc_smd_bin_covar_atc(*args, **kwargs):
    raise NotImplementedError("SMD for ATC estimand is not yet implemented.")


def _calc_smd_cont_covar_att(*args, **kwargs):
    raise NotImplementedError("SMD for ATT estimand is not yet implemented.")


def _calc_smd_cont_covar_atc(*args, **kwargs):
    raise NotImplementedError("SMD for ATC estimand is not yet implemented.")


def _calc_raw_smd(a: float, b: float, std_factor: float) -> float:
    """
    Calculate the raw standardized mean difference (SMD).

    Parameters
    ----------
    a : float
        The mean or weighted mean of the first group.
    b : float
        The mean or weighted mean of the second group.
    std_factor : float
        The standardization factor, typically the pooled standard deviation or other
        relevant standardizing quantity.

    Returns
    -------
    float
        The raw standardized mean difference (SMD).

    Notes
    -----
    The raw SMD is calculated as the absolute difference between `a` and `b` divided by `std_factor`.
    """
    raw_smd = abs(a - b) / std_factor
    return raw_smd
