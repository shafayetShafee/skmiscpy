import os
import sys
from pathlib import Path

def here(path: str) -> str:
    """
    Construct an absolute path relative to the project root directory of the 
    currently active virtual environment.

    parameters:
    -----------
    - path (str): A relative path to be resolved from the project root.

    Returns:
    --------
    str: The absolute path constructed from the project root directory.

    Raises:
    -------
    EnvironmentError: If the script is not running inside a virtual environment.
    """
    if sys.prefix == sys.base_prefix:
        # If they're equal, you're not in a virtual environment, 
        # otherwise you are. Inside a venv, `sys.prefix` points 
        # to the directory of the virtual environment, and 
        # `sys.base_prefix` to the Python interpreter used to 
        # create the environment.
        raise EnvironmentError("Virtual environment is not activated.")
    
    venv_path = Path(os.environ['VIRTUAL_ENV'])
    project_root = venv_path.parent
    # The directory that contains virtual environment will be 
    # considered as the project root directory
    queried_path = Path(path)
    full_abs_path = project_root.joinpath(queried_path)
    return str(full_abs_path)
