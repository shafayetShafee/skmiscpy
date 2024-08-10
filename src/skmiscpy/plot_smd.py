import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union
from .checker import _check_param_type, _check_required_columns


def plot_smd(
    data: pd.DataFrame,
    add_ref_line: bool = False,
    ref_line_value: Union[int, float] = 0.1,
    *args,
    **kwargs,
) -> None:
    """
    Plots the standardized mean difference (SMD) for variables as a point (also known as love-plot),
    displaying unadjusted (and also adjusted, if provided) SMDs. Optionally includes a vertical reference line.

    Parameters:
    -----------
    data : pd.DataFrame, required
        A pandas DataFrame with at least two columns - `variables` and `unadjusted_smd`,
        containing the variables names and their associated unadjusted SMD values. Then
        only the unadjusted SMD will be plotted for the given variables. To include the
        adjusted SMD in the plot, the DataFrame needs to contain another column `adjusted_smd`
        containing the adjusted SMD values for the variables. Note: The column names must
        be `variables`, `unadjusted_smd` and `adjusted_smd`.

    add_ref_line : bool, optional
        Whether to add a vertical reference line. Defaults to False.

    ref_line_value : int or float, optional
        The value at which to draw a vertical reference line. Defaults to 0.1.
        A value between 0 to 1 should be set to this parameter.

    *args : Optional
        Additional positional arguments passed to Seaborn's pointplot.

    **kwargs : Optional
        Additional keyword arguments passed to Seaborn's pointplot.
    """

    _check_param_type({"data": data}, param_type=pd.DataFrame)
    _check_param_type({"add_ref_line": add_ref_line}, param_type=bool)
    _check_param_type({"ref_line_value": ref_line_value}, param_type=(int, float))

    if not (0 <= ref_line_value <= 1):
        raise ValueError("The `ref_line_value` must be between 0 and 1.")

    if data.empty:
        raise ValueError("The input DataFrame is empty. Cannot plot SMD.")

    var_names_col = "variables"
    unadj_smd_col = "unadjusted_smd"
    adj_smd_col = "adjusted_smd"

    _check_required_columns(data, [var_names_col, unadj_smd_col])

    if not pd.api.types.is_numeric_dtype(data[unadj_smd_col]):
        raise TypeError(f"The `{unadj_smd_col}` column must contain numerical data.")

    if adj_smd_col in data.columns and not pd.api.types.is_numeric_dtype(
        data[adj_smd_col]
    ):
        raise TypeError(f"The `{adj_smd_col}` column must contain numerical data.")

    if data[var_names_col].duplicated().any():
        duplicated_vars = data[var_names_col][data[var_names_col].duplicated()].unique()
        raise ValueError(
            f"The `variables` column contains duplicated values: {', '.join(duplicated_vars)}. "
            "Each variable must be unique."
        )

    if adj_smd_col not in data.columns:
        melted_data = data[[var_names_col, unadj_smd_col]].melt(
            id_vars=var_names_col, value_name="SMD", var_name="smd_type"
        )
        melted_data["smd_type"] = "Unadjusted SMD"
    else:
        melted_data = data.melt(
            id_vars=var_names_col,
            value_vars=[unadj_smd_col, adj_smd_col],
            var_name="smd_type",
            value_name="SMD",
        )
        melted_data["smd_type"] = melted_data["smd_type"].replace(
            {unadj_smd_col: "Unadjusted SMD", adj_smd_col: "Adjusted SMD"}
        )

    plt.figure(figsize=(10, 6))

    sns.pointplot(
        data=melted_data, x="SMD", y=var_names_col, hue="smd_type", *args, **kwargs
    )

    if add_ref_line:
        plt.axvline(ref_line_value, color="black", linestyle="--")

    plt.xlabel("Standardized Mean Difference (SMD)")
    plt.ylabel("Variables")
    plt.title("Standardized Mean Difference for Variables")
    plt.legend(title="SMD Type")
    plt.show()
