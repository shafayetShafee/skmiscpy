import os
import sys
from pathlib import Path
from functools import lru_cache


def here(path: str) -> str:
    """
    Construct an absolute path relative to the project root directory.
    Requires an activated virtual environment to determine the project root.

    Parameters:
    -----------
    path : str
        A relative path to be resolved from the project root.

    Returns:
    --------
    str : The absolute path constructed from the project root directory.

    Raises:
    -------
    OSError:
        If the script is not running inside an activated virtual environment
        or if the `VIRTUAL_ENV` environment variable is not set, empty,
        or points to a non-existent directory.
    TypeError:
        If the `path` parameter is not a string.
    ValueError:
        If the `path` parameter is empty or is an absolute path.
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
