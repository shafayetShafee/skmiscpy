import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Union, Optional, Type

def plot_smd(
    data: pd.DataFrame, 
    var_names_col: Optional[str] = None, 
    unadj_smd_col: Optional[str] = None, 
    adj_smd_col: Optional[str] = None, 
    ref_line_value: float = 0.1, 
    add_ref_line: bool = False,
    *args, 
    **kwargs
) -> None:
    """
    Plots the standardized mean difference (SMD) for variables as a point (also known as love-plot), 
    displaying unadjusted (and also adjusted, if provided) SMDs. Optionally includes a vertical reference line.

    Parameters:
    -----------
    - data (pd.DataFrame): The DataFrame containing SMD data.
    - var_names_col (str, optional): The name of column in the `data` that contains the variables for which SMD will be plotted. Defaults to "variable".
    - unadj_smd_col (str, optional): The name of column in the `data` that contains unadjusted SMD. Defaults to "unadjusted_smd".
    - adj_smd_col (str, optional): The name of column in the `data` that contains adjusted SMD. Defaults to "adjusted_smd".
    - ref_line_value (float, optional): The value at which to draw a vertical reference line. Defaults to 0.1.
    - add_ref_line (bool, optional): Whether to add a vertical reference line. Defaults to False.
    - *args: Additional positional arguments passed to Seaborn's pointplot.
    - **kwargs: Additional keyword arguments passed to Seaborn's pointplot.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("The `data` parameter must be a pandas DataFrame.")

    _check_param_type([var_names_col, unadj_smd_col, adj_smd_col], param_type=str)
    _check_param_type(add_ref_line, param_type=bool)

    if not isinstance(ref_line_value, float):
        raise TypeError("The `ref_line_value` must be a numerical value.")

    var_names_col = var_names_col or "variable"
    unadj_smd_col = unadj_smd_col or "unadjusted_smd"
    adj_smd_col = adj_smd_col or "adjusted_smd"

    required_columns = {var_names_col, unadj_smd_col}
    if adj_smd_col in data.columns:
        required_columns.add(adj_smd_col)
    
    if not required_columns.issubset(data.columns):
        missing_cols = required_columns - set(data.columns)
        raise ValueError(f"The DataFrame is missing the following required columns: {', '.join(missing_cols)}")

    if adj_smd_col not in data.columns:
        melted_data = data[[var_names_col, unadj_smd_col]].melt(id_vars=var_names_col, value_name="SMD", var_name="smd_type")
        melted_data["smd_type"] = "Unadjusted SMD"
    else:
        melted_data = data.melt(id_vars=var_names_col, value_vars=[unadj_smd_col, adj_smd_col],
                                var_name="smd_type", value_name="SMD")
        melted_data["smd_type"] = melted_data["smd_type"].replace({unadj_smd_col: "Unadjusted SMD", adj_smd_col: "Adjusted SMD"})

    plt.figure(figsize=(10, 6))

    sns.pointplot(data=melted_data, x="SMD", y=var_names_col, hue="smd_type", *args, **kwargs)

    if add_ref_line:
        plt.axvline(ref_line_value, color='black', linestyle='--')

    plt.xlabel("Standardized Mean Difference (SMD)")
    plt.ylabel("Variables")
    plt.title("Standardized Mean Difference for Variables")
    plt.legend(title='SMD Type')
    plt.show()



def _check_param_type(params: Union[str, List[str]], param_type: Type) -> None:
    """
    Checks if the provided parameter or list of parameters are of the specified type.
    Raises a TypeError if any of them are not of the specified type.

    Parameters:
    -----------
    - params (str, list of str): The parameter or list of parameters to check.
    - param_type (Type): The type to check against (e.g., str, bool).
    """
    if isinstance(params, list):
        for param in params:
            if param is not None and not isinstance(param, param_type):
                raise TypeError(f"The `{param}` parameter must be of type {param_type.__name__}.")
    else:
        if params is not None and not isinstance(params, param_type):
            raise TypeError(f"The `{params}` parameter must be of type {param_type.__name__}.")