import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from skmiscpy import plot_smd


@pytest.fixture(autouse=True)
def no_plot_show(monkeypatch):
    """Fixture to prevent plt.show() from displaying plots during tests."""
    monkeypatch.setattr(plt, "show", lambda: None)


@pytest.fixture
def example_data():
    """Fixture for providing example SMD data."""
    return pd.DataFrame(
        {
            "variables": ["age", "gender", "income", "education"],
            "unadjusted_smd": [0.15, 0.35, 0.25, 0.05],
            "adjusted_smd": [0.08, 0.12, 0.10, 0.02],
        }
    )


def test_plot_smd_invalid_data_type():
    """Test plot_smd with invalid data type for the data parameter."""
    with pytest.raises(TypeError, match="must be of type DataFrame"):
        plot_smd(data=[{"variables": "age", "unadjusted_smd": 0.15}])


def test_plot_smd_missing_columns():
    """Test plotting with missing required columns in DataFrame."""
    df = pd.DataFrame(
        {
            "variables": ["age", "gender", "income", "education"],
        }
    )

    with pytest.raises(
        ValueError, match="The DataFrame is missing the following required columns"
    ):
        plot_smd(data=df)


def test_plot_smd_missing_adjusted_smd_column():
    """Test plotting with only `unadjusted_smd` column present."""
    df = pd.DataFrame(
        {
            "variables": ["age", "gender", "income"],
            "unadjusted_smd": [0.15, 0.35, 0.25],
        }
    )

    try:
        plot_smd(data=df)
    except Exception as e:
        pytest.fail(f"plot_smd raised an exception: {e}")


def test_plot_smd_missing_all_required_columns():
    """Test plotting with missing all required columns in DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})

    with pytest.raises(
        ValueError, match="The DataFrame is missing the following required columns"
    ):
        plot_smd(data=df)


def test_plot_smd_invalid_param_types(example_data):
    """Test plot_smd with invalid parameter types."""
    with pytest.raises(TypeError, match="must be of type int or float"):
        plot_smd(example_data, ref_line_value="high")

    with pytest.raises(TypeError, match="must be of type bool"):
        plot_smd(example_data, add_ref_line="yes")

    with pytest.raises(
        ValueError, match="The `ref_line_value` must be between 0 and 1."
    ):
        plot_smd(example_data, ref_line_value=1.5)


@patch("matplotlib.pyplot.axvline")
def test_plot_smd_basic(mock_axvline, example_data):
    """Test basic functionality of plot_smd."""
    try:
        plot_smd(example_data)
        mock_axvline.assert_not_called()
    except Exception as e:
        pytest.fail(f"plot_smd raised an exception: {e}")


@patch("matplotlib.pyplot.axvline")
def test_plot_smd_with_unadjusted_only(mock_axvline):
    """Test plotting with only 'variables' and 'unadjusted_smd' columns."""
    data = pd.DataFrame(
        {"variables": ["age", "gender", "income"], "unadjusted_smd": [0.15, 0.35, 0.25]}
    )
    try:
        plot_smd(data)
        mock_axvline.assert_not_called()
    except Exception as e:
        pytest.fail(f"plot_smd raised an exception: {e}")


@patch("matplotlib.pyplot.axvline")
def test_plot_smd_with_valid_ref_line(mock_axvline, example_data):
    """Test plotting with a valid reference line value."""
    plot_smd(
        data=example_data,
        add_ref_line=True,
        ref_line_value=0.2,
    )
    mock_axvline.assert_called_once_with(0.2, color="black", linestyle="--")


def test_plot_smd_invalid_ref_line_value(example_data):
    """Test plotting with an invalid reference line value."""
    with pytest.raises(
        ValueError, match="The `ref_line_value` must be between 0 and 1."
    ):
        plot_smd(data=example_data, ref_line_value=1.5, add_ref_line=True)


def test_plot_smd_duplicate_variables():
    """Test plotting with duplicate values in the `variables` column."""
    df = pd.DataFrame(
        {
            "variables": ["age", "gender", "income", "age"],  # Duplicate 'age'
            "unadjusted_smd": [0.15, 0.35, 0.25, 0.05],
            "adjusted_smd": [0.08, 0.12, 0.10, 0.02],
        }
    )

    with pytest.raises(
        ValueError, match="The `variables` column contains duplicated values"
    ):
        plot_smd(data=df)


def test_plot_smd_empty_dataframe():
    """Test plotting with an empty DataFrame."""
    df = pd.DataFrame(columns=["variables", "unadjusted_smd"])

    with pytest.raises(
        ValueError, match="The input DataFrame is empty. Cannot plot SMD."
    ):
        plot_smd(data=df)


def test_plot_smd_extra_columns():
    """Test plotting with unexpected columns in the DataFrame."""
    df = pd.DataFrame(
        {
            "variables": ["age", "gender", "income"],
            "unadjusted_smd": [0.15, 0.35, 0.25],
            "adjusted_smd": [0.08, 0.12, 0.10],
            "extra_column": [1, 2, 3],  # extra column
        }
    )

    try:
        plot_smd(data=df)
    except Exception as e:
        pytest.fail(f"plot_smd raised an exception: {e}")


def test_plot_smd_non_numeric_values_in_smd_columns():
    """Test plotting with non-numeric values in `unadjusted_smd` and `adjusted_smd` columns."""
    df = pd.DataFrame(
        {
            "variables": ["age", "gender", "income"],
            "unadjusted_smd": [0.15, "high", 0.25],  # Non-numeric value
            "adjusted_smd": [0.08, 0.12, "low"],  # Non-numeric value
        }
    )

    with pytest.raises(TypeError, match="must contain numerical data"):
        plot_smd(data=df)
