import pytest
import pandas as pd
import numpy as np

from skmiscpy import compute_smd
from skmiscpy.cbs import _check_smd_data, _calc_smd_covar


@pytest.fixture
def df_bin_zero_variance():
    """Fixture to provide sample data for tests."""
    return pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [1, 1, 0, 0, 1, 0],
            "cont_var": [2.0, 2.5, 3.0, 3.5, 2.2, 3.3],
            "weights": [1.0, 1.5, 2.0, 1.2, 1.1, 0.8],
        }
    )


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for tests."""
    return pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [1, 0, 0, 1, 1, 0],
            "cont_var": [2.0, 2.7, 3.0, 3.5, 2.2, 3.3],
            "weights": [1.0, 1.5, 2.0, 1.2, 1.1, 0.8],
        }
    )


# --- Testing _check_smd_data() ----


def test_check_smd_data_valid(sample_data):
    validated_data = _check_smd_data(
        sample_data, group="group", vars=["binary_var"], wt_var="weights"
    )
    assert isinstance(validated_data, pd.DataFrame)


def test_check_smd_data_invalid_group_type(sample_data):
    with pytest.raises(TypeError, match="The `group` parameter must be of type str"):
        _check_smd_data(sample_data, group=123, vars=["binary_var"], wt_var="weights")


def test_check_smd_data_invalid_vars_type(sample_data):
    with pytest.raises(TypeError, match="`vars` must be a string or a list of strings"):
        _check_smd_data(sample_data, group="group", vars=[123], wt_var="weights")


def test_check_smd_data_missing_group_column(sample_data):
    data_missing_group = sample_data.drop(columns=["group"])
    with pytest.raises(
        ValueError,
        match="The DataFrame is missing the following required columns: group",
    ):
        _check_smd_data(
            data_missing_group, group="group", vars=["binary_var"], wt_var="weights"
        )


def test_check_smd_data_missing_vars_column(sample_data):
    data_missing_var = sample_data.drop(columns=["binary_var"])
    with pytest.raises(
        ValueError,
        match="The DataFrame is missing the following required columns: binary_var",
    ):
        _check_smd_data(
            data_missing_var, group="group", vars=["binary_var"], wt_var="weights"
        )


def test_check_smd_data_missing_values(sample_data):
    data_with_nan = sample_data.copy()
    data_with_nan.loc[0, "binary_var"] = np.nan
    with pytest.raises(
        ValueError, match="The 'binary_var' column contains missing values."
    ):
        _check_smd_data(
            data_with_nan, group="group", vars=["binary_var"], wt_var="weights"
        )


def test_check_smd_data_non_binary_group(sample_data):
    data_non_binary_group = sample_data.copy()
    data_non_binary_group["group"] = [1, 2, 1, 0, 1, 0]
    with pytest.raises(
        ValueError,
        match="The 'group' column must be a binary column for valid SMD calculation",
    ):
        _check_smd_data(
            data_non_binary_group, group="group", vars=["binary_var"], wt_var="weights"
        )


def test_check_smd_data_non_numeric_vars(sample_data):
    data_non_numeric = sample_data.copy()
    data_non_numeric["binary_var"] = ["a", "b", "c", "d", "e", "f"]
    with pytest.raises(ValueError, match="The 'binary_var' column must be numeric"):
        _check_smd_data(
            data_non_numeric, group="group", vars=["binary_var"], wt_var="weights"
        )


def test_check_smd_data_non_positive_weights(sample_data):
    data_non_positive_weights = sample_data.copy()
    data_non_positive_weights["weights"] = [1.0, -1.5, 0, 1.2, 1.1, 0.8]
    with pytest.raises(
        ValueError,
        match="The 'weights' column contains negative weight values. The weight values must be positive",
    ):
        _check_smd_data(
            data_non_positive_weights,
            group="group",
            vars=["binary_var"],
            wt_var="weights",
        )


def test_check_smd_data_non_numeric_binary_conversion(sample_data):
    data_non_numeric_binary = sample_data.copy()
    data_non_numeric_binary["binary_var"] = ["a", "a", "b", "b", "a", "b"]

    expected_conversion = [0, 0, 1, 1, 0, 1]

    validated_data = _check_smd_data(
        data_non_numeric_binary, group="group", vars=["binary_var"], wt_var="weights"
    )
    assert isinstance(validated_data, pd.DataFrame)
    assert "binary_var" in validated_data.columns
    assert all(validated_data["binary_var"] == expected_conversion)
    assert validated_data["binary_var"].dtype == int


# Test --- _calc_smd_covar ----


def test_calc_smd_covar_binary_unweighted(sample_data):
    smd = _calc_smd_covar(data=sample_data, group="group", covar="binary_var")
    assert isinstance(smd, float)
    assert smd > 0


def test_calc_smd_covar_binary_weighted(sample_data):
    smd = _calc_smd_covar(
        data=sample_data, group="group", covar="binary_var", wt_var="weights"
    )
    assert isinstance(smd, float)
    assert smd > 0


def test_calc_smd_covar_continuous_unweighted(sample_data):
    smd = _calc_smd_covar(data=sample_data, group="group", covar="cont_var")
    assert isinstance(smd, float)
    assert smd > 0


def test_calc_smd_covar_continuous_weighted(sample_data):
    smd = _calc_smd_covar(
        data=sample_data, group="group", covar="cont_var", wt_var="weights"
    )
    assert isinstance(smd, float)
    assert smd > 0


# --- zero variance and zero proportion


def test_calc_smd_covar_zero_variance():
    data_zero_variance = pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [1, 1, 0, 0, 1, 0],  # Zero variance for group 1 (all 1s)
            "cont_var": [2.0, 2.5, 3.0, 3.5, 2.2, 3.3],  # Non-zero variance
            "weights": [1.0, 1.5, 2.0, 1.2, 1.1, 0.8],
        }
    )
    with pytest.raises(
        ValueError,
        match="proportion of binary_var for group 1 must be within the range",
    ):
        _calc_smd_covar(
            data_zero_variance, group="group", covar="binary_var", wt_var="weights"
        )

    data_zero_variance_cont = pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [1, 0, 0, 1, 1, 0],  # Non-zero variance
            "cont_var": [
                2.0,
                2.5,
                3.0,
                3.0,
                2.2,
                3.0,
            ],  # Zero variance for group 0 (all 3.0s)
            "weights": [1.0, 1.5, 2.0, 1.2, 1.1, 0.8],
        }
    )
    with pytest.raises(
        ValueError,
        match="The variance of cont_var for group 0 must be strictly positive",
    ):
        _calc_smd_covar(
            data_zero_variance_cont, group="group", covar="cont_var", wt_var="weights"
        )


def test_calc_smd_bin_covar_zero_variance():
    data_zero_proportion = pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [0, 0, 0, 0, 0, 0],  # Zero proportion for group 1 (all 0s)
            "cont_var": [2.0, 2.5, 3.0, 3.5, 2.2, 3.3],
            "weights": [1.0, 1.5, 2.0, 1.2, 1.1, 0.8],
        }
    )
    with pytest.raises(
        ValueError, match="variance of binary_var for group 1 must be strictly positive"
    ):
        _calc_smd_covar(
            data_zero_proportion, group="group", covar="binary_var", wt_var="weights"
        )

    data_zero_proportion_group0 = pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [1, 1, 1, 1, 1, 1],  # Zero proportion for group 0 (all 1s)
            "cont_var": [2.0, 2.5, 3.0, 3.5, 2.2, 3.3],
            "weights": [1.0, 1.5, 2.0, 1.2, 1.1, 0.8],
        }
    )
    with pytest.raises(
        ValueError, match="variance of binary_var for group 1 must be strictly positive"
    ):
        _calc_smd_covar(
            data_zero_proportion_group0,
            group="group",
            covar="binary_var",
            wt_var="weights",
        )


def test_calc_smd_covar_zero_proportion():
    data_zero_proportion = pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [0, 0, 1, 0, 0, 1],  # Zero proportion for group 1 (all 0s)
            "cont_var": [2.0, 2.5, 3.0, 3.5, 2.2, 3.3],
        }
    )
    with pytest.raises(
        ValueError,
        match="proportion of binary_var for group 1 must be within the range",
    ):
        _calc_smd_covar(data_zero_proportion, group="group", covar="binary_var")

    data_zero_proportion_group0 = pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [0, 1, 1, 1, 0, 1],  # Zero proportion for group 0 (all 1s)
            "cont_var": [2.0, 2.5, 3.0, 3.5, 2.2, 3.3],
        }
    )
    with pytest.raises(
        ValueError,
        match="proportion of binary_var for group 0 must be within the range",
    ):
        _calc_smd_covar(data_zero_proportion_group0, group="group", covar="binary_var")


# --- Test compute_smd ----


def test_compute_smd_unweighted(sample_data):
    smd_df = compute_smd(
        data=sample_data, vars=["binary_var", "cont_var"], group="group"
    )
    assert isinstance(smd_df, pd.DataFrame)
    assert "unadjusted_smd" in smd_df.columns
    assert len(smd_df) == 2


def test_compute_smd_weighted(sample_data):
    smd_df = compute_smd(
        data=sample_data,
        vars=["binary_var", "cont_var"],
        group="group",
        wt_var="weights",
    )
    assert isinstance(smd_df, pd.DataFrame)
    assert "unadjusted_smd" in smd_df.columns
    assert "adjusted_smd" in smd_df.columns
    assert len(smd_df) == 2


def test_compute_smd_invalid_group_column(sample_data):
    with pytest.raises(ValueError):
        compute_smd(data=sample_data, vars=["binary_var"], group="invalid_group")


def test_compute_smd_invalid_var_column(sample_data):
    with pytest.raises(ValueError):
        compute_smd(data=sample_data, vars=["invalid_var"], group="group")


def test_compute_smd_invalid_estimand(sample_data):
    with pytest.raises(ValueError):
        compute_smd(
            data=sample_data, vars=["binary_var"], group="group", estimand="INVALID"
        )


def test_compute_smd_att_atc(sample_data):
    att_smd = _calc_smd_covar(
        data=sample_data,
        group="group",
        covar="cont_var",
        wt_var="weights",
        estimand="ATT",
    )
    assert isinstance(att_smd, float), "ATT SMD should return a float value"

    atc_smd = _calc_smd_covar(
        data=sample_data,
        group="group",
        covar="binary_var",
        wt_var="weights",
        estimand="ATC",
    )
    assert isinstance(atc_smd, float), "ATC SMD should return a float value"


# --- Test std_binary param ----

def test_std_binary_calc_smd_covar(sample_data):
    smd = _calc_smd_covar(
        data=sample_data,
        group='group',
        covar='binary_var',
    )
    expected = np.float64(0.3333333)
    np.testing.assert_allclose(smd, expected, rtol=1e-4, atol=0)


def test_compute_smd_invalid_std_binary_type(sample_data):
    with pytest.raises(TypeError, match="The `std_binary` parameter must be of type bool"):
        compute_smd(
            sample_data, 
            group='group', 
            vars=["binary_var"], 
            wt_var="weights",
            std_binary="True"
        )