from typing import Any, Dict, Union, Tuple, Type


def _check_param_type(
    params: Dict[str, Any], 
    param_type: Union[Type, Tuple[Type, ...]]
) -> None:
    """
    Checks if the provided parameters (given as a dictionary of names and values) are of the specified type or types.
    Raises a TypeError if any of them are not of the specified type or types.

    Parameters:
    -----------
    params: Dict[str, Any]
        A dictionary where the key is the parameter name and the value is the parameter value.

    param_type: Type or tuple of Type
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