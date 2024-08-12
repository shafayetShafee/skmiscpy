import os
import sys
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Union, Tuple, Type

import pandas as pd


def here(path: str) -> str:
    """
    Construct an absolute path relative to the project root directory.
    Requires an activated virtual environment to determine the project root.

    Parameters
    ----------
    path : str
        A relative path to be resolved from the project root.

    Returns
    -------
    str
        The absolute path constructed from the project root directory.

    Raises
    ------
    OSError
        If the script is not running inside an activated virtual environment,
        or if the `VIRTUAL_ENV` environment variable is not set, empty,
        or points to a non-existent directory.

    TypeError
        If the `path` parameter is not a string.

    ValueError
        If the `path` parameter is empty or is an absolute path.

    Examples
    --------
    1. Constructing a path to a file in the project:

    >>> from skmiscpy import here
    >>> here("data/input.csv")
    # If the project root is `/home/user/my_project` where you have a virtual env directory,
    # this will return an absolute path like
    # `/home/user/my_project/data/input.csv`.

    2. Constructing a path to a subdirectory:

    >>> here("src/my_module")
    # If the project root is `/home/user/my_project`, this will return an absolute path
    # like `/home/user/my_project/src/my_module`.

    3. Handling errors with an empty path:

    >>> here("")
    # Raises ValueError: The `path` parameter cannot be an empty string.

    4. Handling errors with an absolute path:

    >>> here("/absolute/path/to/file")
    # Raises ValueError: The `path` parameter must be relative, not absolute.

    5. Handling errors with an invalid path type:

    >>> here(123)
    # Raises TypeError: The `path` parameter must be a string.
    """
    if not path:
        raise ValueError("The `path` parameter cannot be an empty string.")

    if not isinstance(path, str):
        raise TypeError("The `path` parameter must be a string.")

    # Expand any environment variables in the path
    expanded_path = os.path.expandvars(path)

    queried_path = Path(expanded_path)

    if queried_path.is_absolute():
        raise ValueError("The `path` parameter must be relative, not absolute.")

    project_root = _get_project_root()
    full_abs_path = (project_root / queried_path).resolve()

    return str(full_abs_path)


@lru_cache(maxsize=1)
def _get_project_root():
    """Cache the project root to avoid repeated lookups."""
    if sys.prefix == sys.base_prefix:
        raise OSError("Virtual environment is not activated.")

    venv_env_var = os.environ.get("VIRTUAL_ENV")
    if not venv_env_var:
        raise OSError(
            "The VIRTUAL_ENV environment variable is not set or is empty. "
            "Ensure that a virtual environment is activated."
        )

    venv_path = Path(venv_env_var)

    if not venv_path.exists():
        raise OSError(
            f"The directory specified by VIRTUAL_ENV ({venv_path}) does not exist."
        )

    return venv_path.parent


def _check_required_columns(
    data: pd.DataFrame, required_columns: Union[str, List[str]]
) -> None:
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
    params: Dict[str, Any], param_type: Union[Type, Tuple[Type, ...]]
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
        return " or ".join(t.__name__ for t in param_type)


def _check_variance_positive(variance: float, custom_msg: str) -> None:
    """
    Check if the variance is strictly positive.

    Parameters
    ----------
    variance : float
        The variance value to check. It should be a positive number.
    var_name : str
        The name of the variable corresponding to the variance. This is used for error messages.

    Raises
    ------
    ValueError
        If the variance is not strictly positive.
    """
    if variance <= 0:
        raise ValueError(
            f"The variance of {custom_msg} must be strictly positive. Found: {variance}."
        )


def _check_proportion_within_range(proportion: float, custom_msg: str) -> None:
    """
    Check if the proportion is within the range [0, 1].

    Parameters
    ----------
    proportion : float
        The proportion value to check. It should be a number between 0 and 1 (inclusive).
    prop_name : str
        The name of the variable corresponding to the proportion. This is used for error messages.

    Raises
    ------
    ValueError
        If the proportion is not within the range [0, 1].
    """
    if not (0 < proportion < 1):
        raise ValueError(
            f"The {custom_msg} must be within the range (0, 1). Found: {proportion}."
        )
