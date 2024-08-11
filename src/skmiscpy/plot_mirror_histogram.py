import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional
from .checker import _check_param_type, _check_required_columns


def plot_mirror_histogram(
    data: pd.DataFrame,
    var: str,
    group: str,
    bins: int = 50,
    weights: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plots a mirror histogram of a variable by another grouping binary variable.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the `var` and `group` column.

    var : str
        Name of the column for which the histogram needs to be drawn.

    group : str
        Name of the binary column based on which the histogram will be mirrored.

    bins : int, optional
        Number of bins for the histograms. Default is 50.

    weights : str, optional
        Name of the column based on which the histogram will be weighted.
        Default is None.

    xlabel : str, optional
        Label for the x-axis. If not provided, defaults to the name of the `var` column.

    ylabel : str, optional
        Label for the y-axis. If not provided, defaults to "Frequency".

    title : str, optional
        Title of the plot. If not provided, defaults to "Mirror Histogram of `var` by `group`".

    Raises
    ------
    TypeError
        If `var`, `group`, `weights`, `xlabel`, `ylabel`, or `title` are not of type `str`.
        If `data` is not a pandas DataFrame.
        If `var` is not numerical.
        If `weights` is not numerical.
    
    ValueError
        If the `bins` parameter is not a positive integer.
        If the `data` DataFrame is empty.
        If the `group` column does not contain exactly two unique, non-NaN values.

    Examples
    --------
    Example 1: Basic usage with numerical data.

    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import numpy as np
    >>> data = pd.DataFrame({
    >>>     'group': [1, 1, 0, 0, 1, 0],
    >>>     'var': [2.0, 3.5, 3.0, 2.2, 2.2, 3.3]
    >>> })
    >>> plot_mirror_histogram(data=data, var='var', group='group')

    Example 2: With weights and custom labels.

    >>> data = pd.DataFrame({
    >>>     'group': [1, 1, 0, 0, 1, 0],
    >>>     'var': [2.0, 3.5, 3.0, 2.2, 2.2, 3.3],
    >>>     'weights': [1.0, 1.5, 2.0, 1.2, 1.1, 0.8]
    >>> })
    >>> plot_mirror_histogram(
    >>>     data=data, var='var', group='group', weights='weights',
    >>>     xlabel='Variable', ylabel='Count', title='Weighted Mirror Histogram'
    >>> )
    """

    _check_param_type({"data": data}, pd.DataFrame)
    _check_param_type({"var": var, "group": group}, str)

    if bins is None:
        bins = 50
    else:
        _check_param_type({"bins": bins}, int)
        if bins <= 0:
            raise ValueError("The `bins` parameter must be a positive integer.")

    if xlabel is not None:
        _check_param_type({"xlabel": xlabel}, str)

    if ylabel is not None:
        _check_param_type({"ylabel": ylabel}, str)

    if title is not None:
        _check_param_type({"title": title}, str)

    if data.empty:
        raise ValueError("The input DataFrame is empty. Cannot plot histogram.")

    required_columns = [var, group]

    if weights is not None:
        _check_param_type({"weights": weights}, str)
        required_columns.append(weights)

    _check_required_columns(data, required_columns)

    unique_groups = data[group].unique()
    if len(unique_groups) != 2 or pd.isna(unique_groups).any():
        raise ValueError(
            "The grouping variable must have exactly two unique non-NaN values."
        )

    if not np.issubdtype(data[var].dtype, np.number):
        raise TypeError(f"The `{var}` column must contain numerical data.")

    if weights is not None and not pd.api.types.is_numeric_dtype(data[weights]):
        raise TypeError(f"The `{weights}` column must contain numerical data.")

    group1, group2 = unique_groups

    if weights:
        weights_group1 = data.query(f"{group} == @group1")[weights]
        weights_group2 = data.query(f"{group} == @group2")[weights]
    else:
        weights_group1 = None
        weights_group2 = None

    sns.histplot(
        x=data.query(f"{group} == @group1")[var],
        bins=bins,
        weights=weights_group1,
        edgecolor="white",
        color="#0072B2",
        label=f"Group {group1}",
    )

    heights, bins = np.histogram(
        a=data.query(f"{group} == @group2")[var], bins=bins, weights=weights_group2
    )
    heights *= -1  # Reverse the heights for the second group
    bin_width = np.diff(bins)[0]
    bin_pos = bins[:-1] + bin_width / 2

    plt.bar(
        bin_pos,
        heights,
        width=bin_width,
        edgecolor="white",
        color="#D55E00",
        label=f"Group {group2}",
    )

    # Adjust y-axis to show positive values for both groups
    ticks = plt.gca().get_yticks()
    plt.gca().set_yticks(ticks)
    plt.gca().set_yticklabels([abs(int(tick)) for tick in ticks])

    if xlabel is None:
        xlabel = f"{var}"
    if ylabel is None:
        ylabel = "Frequency"
    if title is None:
        title = f"Mirror Histogram of {var} by {group}"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
