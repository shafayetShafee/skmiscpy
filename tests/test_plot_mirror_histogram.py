import pytest
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skmiscpy import plot_mirror_histogram

@pytest.fixture(autouse=True)
def no_plot_show(monkeypatch):
    """Fixture to prevent plt.show() from displaying plots during tests."""
    monkeypatch.setattr(plt, "show", lambda: None)


@pytest.fixture
def sample_data():
    """Fixture to provide a sample DataFrame for testing."""
    data = pd.DataFrame(
        {
            "variable": np.random.randn(100),
            "group": np.random.choice(["A", "B"], size=100),
            "weights": np.random.rand(100),
        }
    )
    return data


def test_plot_mirror_histogram_valid_data(sample_data):
    """Test plot_mirror_histogram with valid parameters and data."""
    try:
        plot_mirror_histogram(
            data=sample_data,
            var="variable",
            group="group",
            bins=30,
            weights="weights",
            xlabel="Variable",
            ylabel="Frequency",
            title="Sample Histogram",
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_plot_mirror_histogram_missing_columns(sample_data):
    """Test plot_mirror_histogram with missing columns in data."""
    data = sample_data.drop(columns="weights")
    with pytest.raises(
        ValueError,
        match="The DataFrame is missing the following required columns: weights",
    ):
        plot_mirror_histogram(
            data=data, var="variable", group="group", weights="weights"
        )


def test_plot_mirror_histogram_invalid_bins(sample_data):
    """Test plot_mirror_histogram with invalid bins parameter."""
    with pytest.raises(
        ValueError, match="The `bins` parameter must be a positive integer."
    ):
        plot_mirror_histogram(data=sample_data, var="variable", group="group", bins=-10)


def test_plot_mirror_histogram_non_numeric_var(sample_data):
    """Test plot_mirror_histogram with non-numeric data for var."""
    data = sample_data.copy()
    data["variable"] = data["variable"].astype(str)  # Convert to non-numeric
    with pytest.raises(
        TypeError, match="The `variable` column must contain numerical data."
    ):
        plot_mirror_histogram(data=data, var="variable", group="group")


def test_plot_mirror_histogram_non_numeric_weights(sample_data):
    """Test plot_mirror_histogram with non-numeric data for weights."""
    data = sample_data.copy()
    data["weights"] = data["weights"].astype(str)  # Convert to non-numeric
    with pytest.raises(
        TypeError, match="The `weights` column must contain numerical data."
    ):
        plot_mirror_histogram(
            data=data, var="variable", group="group", weights="weights"
        )


def test_plot_mirror_histogram_empty_dataframe():
    """Test plot_mirror_histogram with an empty DataFrame."""
    empty_data = pd.DataFrame(columns=["variable", "group", "weights"])
    with pytest.raises(
        ValueError, match="The input DataFrame is empty. Cannot plot histogram."
    ):
        plot_mirror_histogram(
            data=empty_data, var="variable", group="group", weights="weights"
        )


def test_plot_mirror_histogram_non_binary_group(sample_data):
    """Test plot_mirror_histogram with a non-binary group variable."""
    data = sample_data.copy()
    data["group"] = np.random.choice(["A", "B", "C"], size=100)  # Add a third group
    with pytest.raises(
        ValueError,
        match="The grouping variable must have exactly two unique non-NaN values.",
    ):
        plot_mirror_histogram(data=data, var="variable", group="group")
