import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Union, Optional, Type


def compute_smd(
    data: pd.DataFrame,
    vars: List[str],
    group: str,
    wt_var: str = None,
    estimand: str = "ATE",
) -> pd.DataFrame:
    """
    Computes the standardized mean difference (SMD) for a list of variables.

    Parameters:
    -----------
    - data (pd.DataFrame): A pandas DataFrame containing the `vars`, `group`, and `wt_var` columns.
    - vars (List[str]): A list of strings representing the variables for which to calculate the SMD.
    - group (str): The name of the binary group column (e.g., treatment vs. control).
    - wt_var (str, optional): The name of the weights column. Defaults to None.
    - estimand (str, optional): The estimand type. Only 'ATE' is supported. Defaults to 'ATE'.

    Returns:
    --------
    pd.DataFrame: A DataFrame with two columns: 'Variable' and 'SMD', representing the variable names and their corresponding SMD values.
    """

    data = _check_smd_data(
        data=data, group=group, vars=vars, wt_var=wt_var, estimand=estimand
    )

    smd_results = []

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
                    "variable": var,
                    "unadjusted_smd": unadjusted_smd,
                    "adjusted_smd": adjusted_smd,
                }
            )
        else:
            unadjusted_smd = _calc_smd_covar(
                data=data, group=group, covar=var, estimand=estimand
            )
            smd_results.append({"variable": var, "unadjusted_smd": unadjusted_smd})

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
    Calculates standardized mean differences (SMD) of a covariate for the `group` variable.

    Parameters:
    -----------
    - data (pd.DataFrame): The input data containing the group, covariate, and optional weights.
    - group (str): The column name indicating the group. This column must be a binary column.
    - covar (str): The column name of the covariate for which the SMD is calculated.
    - wt_var (str, optional): The column name of the weights. If None, only unadjusted SMD is calculated. Defaults to None.
    - estimand (str, optional): The causal estimand to use. Supports only ATE now.

    Returns:
    --------
    - float: Standardized Mean Difference.
    """
    data = _check_smd_data(
        data=data, group=group, vars=covar, wt_var=wt_var, estimand=estimand
    )

    grp_1_dt = data[data[group] == 1]
    grp_0_dt = data[data[group] == 0]
    covar_1 = grp_1_dt[covar]
    covar_0 = grp_0_dt[covar]

    m1 = covar_1.mean()
    m0 = covar_0.mean()
    s2_1 = covar_1.var()
    s2_0 = covar_0.var()

    if wt_var is not None:
        wt_m1 = np.average(covar_1, weights=grp_1_dt[wt_var])
        wt_m0 = np.average(covar_0, weights=grp_0_dt[wt_var])
        if data[covar].dropna().nunique() == 2:
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
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The `data` parameter must be a pandas DataFrame.")

    vars = [vars] if isinstance(vars, str) else vars

    required_columns = set(vars + [group])
    if wt_var is not None:
        required_columns.add(wt_var)

    if not required_columns.issubset(data.columns):
        missing_cols = required_columns - set(data.columns)
        raise ValueError(
            f"The DataFrame is missing the following required columns: {', '.join(missing_cols)}"
        )

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
            min_val, max_val = min(unique_vals), max(unique_vals)
            data[var] = data[var].apply(lambda x: 0 if x == min_val else 1)

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


def _calc_smd_bin_covar_ate(m1, m0, wt_m1=None, wt_m0=None):
    wt_m1 = m1 if wt_m1 is None else wt_m1
    wt_m0 = m0 if wt_m0 is None else wt_m0
    pooled_var = (m1 * (1 - m1)) + (m0 * (1 - m0))
    std_factor = np.sqrt(pooled_var / 2)
    smd = _calc_raw_smd(a=wt_m1, b=wt_m0, std_factor=std_factor)
    return smd


def _calc_smd_bin_covar_att(*args, **kwargs):
    raise NotImplementedError("SMD for ATT is not yet implemented.")


def _calc_smd_bin_covar_atc(*args, **kwargs):
    raise NotImplementedError("SMD for ATC is not yet implemented.")


def _calc_smd_cont_covar_ate(m1, m0, s2_1, s2_0):
    std_factor = np.sqrt((s2_1 + s2_0) / 2)
    smd = _calc_raw_smd(a=m1, b=m0, std_factor=std_factor)
    return smd


def _calc_smd_cont_covar_att(*args, **kwargs):
    raise NotImplementedError("SMD for ATT is not yet implemented.")


def _calc_smd_cont_covar_atc(*args, **kwargs):
    raise NotImplementedError("SMD for ATC is not yet implemented.")


def _calc_raw_smd(a, b, std_factor):
    raw_smd = abs(a - b) / std_factor
    return raw_smd
