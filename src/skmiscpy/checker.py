import pandas as pd
from typing import Any, Dict, List, Union, Tuple, Type

def _check_required_columns(data: pd.DataFrame, required_columns: Union[str, List[str]]) -> None:
    """
    Checks if the DataFrame contains the required columns.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to check for required columns.

    required_columns : str or list of str
        A column name or a list of column names that are required to be present in the DataFrame.

    Raises
    ------
    ValueError
        If any of the required columns are missing from the DataFrame.
    """
    if isinstance(required_columns, str):
        required_columns = {required_columns}
    elif isinstance(required_columns, list):
        required_columns = set(required_columns)
    else:
        raise TypeError("`required_columns` must be a string or a list of strings.")
    
    if not required_columns.issubset(data.columns):
        missing_cols = required_columns - set(data.columns)
        raise ValueError(
            f"The DataFrame is missing the following required columns: {', '.join(missing_cols)}"
        )



def _check_param_type(
    params: Dict[str, Any], 
    param_type: Union[Type, Tuple[Type, ...]]
) -> None:
    """
    Checks if the provided parameters (given as a dictionary of names and values) are of the specified type or types.
    Raises a TypeError if any of them are not of the specified type or types.

    Parameters:
    -----------
    params : Dict[str, Any]
        A dictionary where the key is the parameter name and the value is the parameter value.

    param_type : Type or tuple of Type
        The type or tuple of types to check against (e.g., str, bool, (int, float)).
    """
    for param_name, param_value in params.items():
        if param_value is not None and not isinstance(param_value, param_type):
            raise TypeError(
                f"The `{param_name}` parameter must be of type {_get_type_name(param_type)}."
            )


def _get_type_name(param_type: Union[Type, Tuple[Type, ...]]) -> str:
    """Get the type name in string"""
    if isinstance(param_type, Type):
        return param_type.__name__
    else:
        return ' or '.join(t.__name__ for t in param_type)