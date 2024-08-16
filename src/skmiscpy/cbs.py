import numpy as np
import pandas as pd

import warnings
from typing import List
from skmiscpy.utils import _check_param_type, _check_required_columns, _classify_columns
from skmiscpy.utils import _check_variance_positive, _check_proportion_within_range


def compute_smd(
    data: pd.DataFrame,
    vars: List[str],
    group: str,
    wt_var: str = None,
    cat_vars: List[str] = None,
    std_binary: bool = False,
    estimand: str = "ATE",
) -> pd.DataFrame:
    """
    Computes the standardized mean difference (SMD) for a list of variables.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the columns specified in ``vars``, ``group``, and optionally ``wt_var``.

    vars : List[str]
        A list of strings representing the variables names for which to calculate the SMD, where the
        variables should be either continuous or binary. The values of the binary variable could be
        either string type or numerical, they would be converted into 0 and 1 (if they are not already
        0-1), where lower value converted into 0 and higher value converted into 1. To compute SMD for
        a discrete variable with more than two categories, pass that variable name in a list to the
        ``cat_vars`` parameter.

    group : str
        The name of the binary group column based on which the mean differences will be calculated.

    wt_var : str, optional
        The name of the column containing weights. Defaults to None.

    cat_vars: List[str], optional
        A list of strings representing the categorical (i.e. discrete) variables among the list
        specified in the ``vars`` parameter.

    std_binary: bool
        Should the mean differences for binary variables (i.e., difference in proportion)
        be standardized or not. Default is False. See notes.

    estimand : str, optional
        The estimand type. Currently, only ``"ATE"`` (Average Treatment Effect) is supported. Defaults to ``"ATE"``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:

        * ``variables``: The name of the variable.
        * ``var_types``: The type of the variable (Continuous or Binary).
        * ``unadjusted_smd``: The standardized mean difference without adjustment.
        * ``adjusted_smd``: The standardized mean difference with adjustment (if ``wt_var`` is provided).

    Notes
    -----

    The mean differences for continuous variables are standardized so that they are on the same scale
    and so that they can be compared across variables, and they allow for a simple interpretation even
    when the details of the variable's original scale are unclear to the analyst.

    None of these advantages are passed to binary variables because binary variables are already on the
    same scale (i.e., a proportion), and the scale is easily interpretable. In addition, the details of
    standardizing the proportion difference of a binary variable involve dividing the proportion difference
    by a variance, but the variance of a binary variable is a function of its proportion. Standardizing
    the proportion difference of a binary variable can yield the following counterintuitive result:
    if P\ :sub:`T`\ = 0.2 and P\ :sub:`C`\ = 0.3, the standardized difference in proportion would be
    different from that if P\ :sub:`T`\ = 0.5 and P\ :sub:`C`\ = 0.6, even though the expectation is that
    the balance statistic should be the same for both scenarios because both would yield the same degree
    of bias in the effect estimate. If still you want the standardized mean difference for binary variables,
    use ``std_binary = True`` in ``compute_smd()``.


    Examples
    --------
    >>> import pandas as pd
    >>> from skmiscpy import compute_smd
    >>> import numpy as np

    >>> sample_df = pd.DataFrame({
    ...     'age': np.random.randint(18, 66, size=100),
    ...     'weight': np.round(np.random.uniform(120, 200, size=100), 1),
    ...     'gender': np.random.choice(['male', 'female'], size=100),
    ...     'race': np.random.choice(
    ...         ['white', 'black', 'hispanic'],
    ...         size=100, p=[0.4, 0.3, 0.3]
    ...     ),
    ...     'educ_level': np.random.choice(
    ...         ['bachelor', 'master', 'doctorate'],
    ...         size=100, p=[0.3, 0.4, 0.3]
    ...     ),
    ...     'ps_wts': np.round(np.random.uniform(0.1, 1.0, size=100), 2),
    ...     'group': np.random.choice(['treated', 'control'], size=100),
    ...     'date': pd.date_range(start='2024-01-01', periods=100, freq='D')
    ... })

    1. Basic usage with unadjusted SMD only:

    >>> compute_smd(sample_df, vars=['age', 'weight', 'gender'], group='group')
    # Returns a DataFrame with unadjusted SMD values for 'age' and 'weight'.

    2. Including weights for adjusted SMD:

    >>> compute_smd(sample_df, vars=['age', 'weight', 'gender'], group='group', wt_var='ps_wts')
    # Returns a DataFrame with both unadjusted and adjusted SMD values for 'age' and 'weight'.

    3. Including categorical variables for adjusted SMD:

    >>> compute_smd(
    ...     sample_df,
    ...     vars=['age', 'weight', 'gender'],
    ...     group='group',
    ...     wt_var='ps_wts',
    ...     cat_vars=['race', 'educ_level']
    ... )
    # Returns a DataFrame with unadjusted and adjusted SMD values for 'age', 'weight', 'race', and 'educ_level'.

    """
    _check_param_type({"data": data}, pd.DataFrame)
    _check_param_type({"group": group}, str)
    _check_param_type({"std_binary": std_binary}, bool)

    if estimand is not None:
        _check_param_type({"estimand": estimand}, str)
    else:
        warnings.warn(
            "Estimand can not be None. Results are shown considering 'ATE' as the estimand.",
            UserWarning,
        )
        estimand = "ATE"

    if not (isinstance(vars, list) and all(isinstance(v, str) for v in vars)):
        raise TypeError("`vars` must be a list of strings")

    if wt_var is not None:
        _check_param_type({"wt_var": wt_var}, str)

    if cat_vars is not None:
        if not (
            isinstance(cat_vars, list) and all(isinstance(v, str) for v in cat_vars)
        ):
            raise TypeError("`cat_vars` must be a list of strings")

    data = _check_prep_smd_data(
        data=data, group=group, vars=vars, wt_var=wt_var, cat_vars=cat_vars
    )

    covariates = list(set(data.columns) - {wt_var, group})
    covariates_with_types = _classify_columns(data, covariates)

    if not std_binary:
        if any(col_type == "binary" for col_type in covariates_with_types.values()):
            print(
                "For binary variables, the unstandardized mean differences are shown here. "
                "See 'Notes' in function documentation for details."
            )

    smd_results = []

    for var, var_type in covariates_with_types.items():
        if wt_var is not None:
            unadjusted_smd = _calc_smd_covar(
                data=data,
                group=group,
                covar=var,
                estimand=estimand,
                std_binary=std_binary,
            )
            adjusted_smd = _calc_smd_covar(
                data=data,
                group=group,
                covar=var,
                wt_var=wt_var,
                estimand=estimand,
                std_binary=std_binary,
            )
            smd_results.append(
                {
                    "variables": var,
                    "var_types": var_type,
                    "unadjusted_smd": unadjusted_smd,
                    "adjusted_smd": adjusted_smd,
                }
            )
        else:
            unadjusted_smd = _calc_smd_covar(
                data=data,
                group=group,
                covar=var,
                estimand=estimand,
                std_binary=std_binary,
            )
            smd_results.append(
                {
                    "variables": var,
                    "var_types": var_type,
                    "unadjusted_smd": unadjusted_smd,
                }
            )

    smd_df = pd.DataFrame(smd_results)

    return smd_df.sort_values(by=["var_types", "variables"], ascending=[False, True])


def _check_prep_smd_data(
    data: pd.DataFrame,
    group: str,
    vars: List[str],
    wt_var: str = None,
    cat_vars: List[str] = None,
) -> pd.DataFrame:
    """
    Validate and prepare the input data for SMD calculation.

    Raises
    ------
    TypeError:
        If data params is not a pd.DataFrame, if group, covar, or wt_var params are not strings.
    ValueError
        if estimand is invalid, if the group column does not contain binary values, or if the weight column
        has non-positive or zero values.
    """
    required_columns = list(
        set(
            vars
            + [group]
            + ([wt_var] if wt_var else [])
            + (cat_vars if cat_vars else [])
        )
    )
    _check_required_columns(data, required_columns)
    data = data.loc[:, required_columns]

    for col in required_columns:
        if data[col].isnull().any():
            raise ValueError(f"The '{col}' column contains missing values.")

    if wt_var is not None:
        if not pd.api.types.is_numeric_dtype(data[wt_var]):
            raise ValueError(f"The '{wt_var}' column must be numeric.")

        if (data[wt_var] <= 0).any():
            raise ValueError(
                f"The '{wt_var}' column contains negative weight values. The weight values must be positive"
            )

    unique_groups = data[group].dropna().unique()
    if len(unique_groups) == 2:
        if set(unique_groups) != {0, 1}:
            data[group] = (data[group] == unique_groups.max()).astype(int)
    else:
        raise ValueError(
            f"The '{group}' column must be a binary column for valid SMD calculation."
        )

    binary_vars = {}
    cat_vars_to_dummy = []
    cols_to_iterate = set(vars + (cat_vars if cat_vars else []))

    for var in cols_to_iterate:
        unique_vals = data[var].dropna().unique()
        if len(unique_vals) == 2:
            binary_vars[var] = unique_vals
        elif cat_vars and var in cat_vars and len(unique_vals) > 2:
            cat_vars_to_dummy.append(var)
        elif not pd.api.types.is_numeric_dtype(data[var]):
            raise ValueError(
                f"The '{var}' column must be continuous or binary."
                "if it is a categorical column with more than two category,"
                "enlist the column name using `cat_vars` parameter."
            )

    for bin_var, bin_val in binary_vars.items():
        data[bin_var] = (data[bin_var] == bin_val.max()).astype(int)

    if cat_vars_to_dummy:
        data = pd.get_dummies(
            data, columns=cat_vars_to_dummy, prefix=cat_vars_to_dummy, dtype=int
        )

    return data


def _calc_smd_covar(
    data: pd.DataFrame,
    group: str,
    covar: str,
    wt_var: str = None,
    estimand: str = "ATE",
    std_binary: bool = False,
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
        The causal estimand to use. Defaults to "ATE" (Average Treatment Effect). Currently supported
        options are "ATT" (Average Treatment Effect among the Treated) and
        "ATC" (Average Treatment Effect among the Control group).
    std_binary: bool
        Should the mean differences for binary variables (i.e., difference in proportion)
        be standardized or not. Default is False. See notes.

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
            return _calc_smd_bin_covar(
                estimand, m1=m1, m0=m0, wt_m1=wt_m1, wt_m0=wt_m0, std_binary=std_binary
            )
        else:
            return _calc_smd_cont_covar(
                estimand, m1=wt_m1, m0=wt_m0, s2_1=s2_1, s2_0=s2_0
            )
    else:
        if data[covar].dropna().nunique() == 2:
            return _calc_smd_bin_covar(estimand, m1=m1, m0=m0, std_binary=std_binary)
        else:
            return _calc_smd_cont_covar(estimand, m1=m1, m0=m0, s2_1=s2_1, s2_0=s2_0)


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
    m1: float,
    m0: float,
    wt_m1: float = None,
    wt_m0: float = None,
    std_binary: bool = False,
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
    std_factor = np.sqrt(pooled_var / 2) if std_binary else 1

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


def _calc_smd_bin_covar_att(
    m1: float,
    m0: float,
    wt_m1: float = None,
    wt_m0: float = None,
    std_binary: bool = False,
) -> float:
    """
    Calculate the standardized mean difference (SMD) for binary covariates
    when estimand is the Average Treatment Effect among the Treated group (ATT).

    Parameters
    ----------
    m1 : float
        The mean of the covariate for the treatment group. Must be between 0 and 1.
    m0 : float
        The mean of the covariate for the control group. Must be between 0 and 1.
    wt_m1 : float, optional
        The weighted mean of the covariate for the treatment group.
        If not provided, `m1` is used. Must be between 0 and 1.
    wt_m0 : float, optional
        The weighted mean of the covariate for the control group. I
        f not provided, `m0` is used. Must be between 0 and 1.

    Returns
    -------
    float
        The Standardized Mean Difference (SMD).
    """
    wt_m1 = m1 if wt_m1 is None else wt_m1
    wt_m0 = m0 if wt_m0 is None else wt_m0

    std_factor = np.sqrt(m1 * (1 - m1)) if std_binary else 1

    smd = _calc_raw_smd(a=wt_m1, b=wt_m0, std_factor=std_factor)
    return smd


def _calc_smd_bin_covar_atc(
    m1: float,
    m0: float,
    wt_m1: float = None,
    wt_m0: float = None,
    std_binary: bool = False,
) -> float:
    """
    Calculate the standardized mean difference (SMD) for binary covariates
    when estimand is the Average Treatment Effect among the Control group (ATC).

    Parameters
    ----------
    m1 : float
        The mean of the covariate for the treatment group. Must be between 0 and 1.
    m0 : float
        The mean of the covariate for the control group. Must be between 0 and 1.
    wt_m1 : float, optional
        The weighted mean of the covariate for the treatment group.
        If not provided, `m1` is used. Must be between 0 and 1.
    wt_m0 : float, optional
        The weighted mean of the covariate for the control group. I
        f not provided, `m0` is used. Must be between 0 and 1.

    Returns
    -------
    float
        The Standardized Mean Difference (SMD).
    """
    wt_m1 = m1 if wt_m1 is None else wt_m1
    wt_m0 = m0 if wt_m0 is None else wt_m0

    std_factor = np.sqrt(m0 * (1 - m0)) if std_binary else 1

    smd = _calc_raw_smd(a=wt_m1, b=wt_m0, std_factor=std_factor)
    return smd


def _calc_smd_cont_covar_att(m1: float, m0: float, s2_1: float, s2_0: float) -> float:
    """
    Calculate the standardized mean difference (SMD) for continuous covariates
    when estimand is the Average Treatment Effect among the Treated group (ATT).

    Parameters
    ----------
    m1 : float
        The mean of the covariate for treated group (group 1).
    m0 : float
        The mean of the covariate for control group (group 0).
    s2_1 : float
        The variance of the covariate for treated group (group 1).
        Must be strictly positive.
    s2_0 : float
        The variance of the covariate for control group (group 0).
        Must be strictly positive.

    Returns
    -------
    float
        The standardized mean difference (SMD).
    """
    std_factor = np.sqrt(s2_1)
    smd = _calc_raw_smd(a=m1, b=m0, std_factor=std_factor)
    return smd


def _calc_smd_cont_covar_atc(m1: float, m0: float, s2_1: float, s2_0: float) -> float:
    """
    Calculate the standardized mean difference (SMD) for continuous covariates
    when estimand is the Average Treatment Effect among the Control group (ATC).

    Parameters
    ----------
    m1 : float
        The mean of the covariate for treated group (group 1).
    m0 : float
        The mean of the covariate for control group (group 0).
    s2_1 : float
        The variance of the covariate for treated group (group 1).
        Must be strictly positive.
    s2_0 : float
        The variance of the covariate for control group (group 0).
        Must be strictly positive.

    Returns
    -------
    float
        The standardized mean difference (SMD).
    """
    std_factor = np.sqrt(s2_0)
    smd = _calc_raw_smd(a=m1, b=m0, std_factor=std_factor)
    return smd


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
