import pytest
import pandas as pd
import numpy as np

from skmiscpy import compute_smd
from skmiscpy.cbs import _check_prep_smd_data, _calc_smd_covar


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'weight': [150.5, 160.0, 155.3, 165.2, 170.8],
        'gender_binary': [0, 1, 0, 1, 0],
        'gender_label': ['male', 'female', 'male', 'female', 'male'],
        'race': ['white', 'black', 'hispanic', 'white', 'black'],
        'educ_level': ['bachelor', 'master', 'doctorate', 'bachelor', 'master'],
        'ps_wts': [0.2, 0.4, 0.6, 0.8, 1.0],
        'group': ['treated', 'control', 'treated', 'control', 'treated'],
        'date': pd.date_range(start='2024-01-01', periods=5, freq='D')
    })

@pytest.fixture
def small_sample_data():
    """Fixture to provide sample data for tests."""
    return pd.DataFrame(
        {
            "group": [1, 1, 0, 0, 1, 0],
            "binary_var": [1, 0, 0, 1, 1, 0],
            "cont_var": [2.0, 2.7, 3.0, 3.5, 2.2, 3.3],
            "weights": [1.0, 1.5, 2.0, 1.2, 1.1, 0.8],
        }
    )

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


# --- Test std_binary param ----------------------------------------------------------

def test_std_binary_calc_smd_covar(small_sample_data):
    smd = _calc_smd_covar(
        data=small_sample_data,
        group="group",
        covar="binary_var",
    )
    expected = np.float64(0.3333333)
    np.testing.assert_allclose(smd, expected, rtol=1e-4, atol=0)


def test_compute_smd_invalid_std_binary_type(small_sample_data):
    with pytest.raises(
        TypeError, match="The `std_binary` parameter must be of type bool"
    ):
        compute_smd(
            small_sample_data,
            group="group",
            vars=["binary_var"],
            wt_var="weights",
            std_binary="True",
        )


# Testing _check_prep_smd_data() -----------------------------------------------------

def test_check_prep_smd_data_transformations_1(sample_data):
    transformed_data = _check_prep_smd_data(
        sample_data,
        group='group',
        vars=['age', 'weight', 'gender_binary', 'gender_label', 'race', 'educ_level'],
        wt_var='ps_wts',
        cat_vars=['race', 'educ_level']
    )

    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape[1] == 12
    assert transformed_data['race_black'].equals(pd.Series([0, 1, 0, 0, 1], name='race_black'))
    assert transformed_data['race_hispanic'].equals(pd.Series([0, 0, 1, 0, 0], name='race_hispanic'))
    assert transformed_data['race_white'].equals(pd.Series([1, 0, 0, 1, 0], name='race_white'))
    assert transformed_data['educ_level_bachelor'].equals(pd.Series([1, 0, 0, 1, 0], name='educ_level_bachelor'))
    assert transformed_data['educ_level_doctorate'].equals(pd.Series([0, 0, 1, 0, 0], name='educ_level_doctorate'))
    assert transformed_data['educ_level_master'].equals(pd.Series([0, 1, 0, 0, 1], name='educ_level_master'))

    pd.testing.assert_series_equal(
        transformed_data['age'],
        sample_data['age'],
        check_index_type=False,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        transformed_data['weight'],
        sample_data['weight'],
        check_index_type=False,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        transformed_data['ps_wts'],
        sample_data['ps_wts'],
        check_index_type=False,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        transformed_data['gender_binary'],
        sample_data['gender_binary'],
        check_index_type=False,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        transformed_data['gender_label'],
        pd.Series([1, 0, 1, 0, 1], name='gender_label'),
        check_index_type=False,
        check_dtype=False
    )


def test_check_prep_smd_data_transformations_2(sample_data):
    transformed_data = _check_prep_smd_data(
        sample_data,
        group='group',
        vars=['age', 'weight', 'gender_binary', 'gender_label'],
        cat_vars=['race', 'educ_level']
    )

    assert transformed_data.shape[1] == 11
    assert transformed_data['race_black'].equals(pd.Series([0, 1, 0, 0, 1], name='race_black'))
    assert transformed_data['race_hispanic'].equals(pd.Series([0, 0, 1, 0, 0], name='race_hispanic'))
    assert transformed_data['race_white'].equals(pd.Series([1, 0, 0, 1, 0], name='race_white'))
    assert transformed_data['educ_level_bachelor'].equals(pd.Series([1, 0, 0, 1, 0], name='educ_level_bachelor'))
    assert transformed_data['educ_level_doctorate'].equals(pd.Series([0, 0, 1, 0, 0], name='educ_level_doctorate'))
    assert transformed_data['educ_level_master'].equals(pd.Series([0, 1, 0, 0, 1], name='educ_level_master'))

    pd.testing.assert_series_equal(
        transformed_data['age'],
        sample_data['age'],
        check_index_type=False,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        transformed_data['weight'],
        sample_data['weight'],
        check_index_type=False,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        transformed_data['gender_binary'],
        sample_data['gender_binary'],
        check_index_type=False,
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        transformed_data['gender_label'],
        pd.Series([1, 0, 1, 0, 1], name='gender_label'),
        check_index_type=False,
        check_dtype=False
    )

def test_check_prep_smd_data_missing_column(sample_data):
    with pytest.raises(ValueError, match="The DataFrame is missing the following required columns"):
        _check_prep_smd_data(
            sample_data,
            group='group',
            vars=['age', 'nonexistent'],
            wt_var='ps_wts'
        )

def test_check_prep_smd_data_invalid_weight(sample_data):
    invalid_data = sample_data.copy()
    invalid_data['ps_wts'] = [-0.2, 0.4, -0.6, 0.8, 1.0]
    with pytest.raises(ValueError, match="The 'ps_wts' column contains negative weight values."):
        _check_prep_smd_data(
            invalid_data,
            group='group',
            vars=['age', 'gender_label', 'race'],
            wt_var='ps_wts'
        )

def test_check_prep_smd_data_non_numeric_weight(sample_data):
    invalid_data = sample_data.copy()
    invalid_data['ps_wts'] = ['a', 'b', 'c', 'd', 'e']
    with pytest.raises(ValueError, match="The 'ps_wts' column must be numeric."):
        _check_prep_smd_data(
            invalid_data,
            group='group',
            vars=['age', 'weight', 'gender_binary', 'gender_label', 'race'],
            wt_var='ps_wts'
        )

def test_check_prep_smd_data_non_binary_group(sample_data):
    invalid_data = sample_data.copy()
    invalid_data['group'] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="The 'group' column must be a binary column for valid SMD calculation."):
        _check_prep_smd_data(
            invalid_data,
            group='group',
            vars=['age', 'gender_binary', 'race']
        )

def test_check_prep_smd_data_non_numeric_or_categorical(sample_data):
    with pytest.raises(ValueError, match="The 'date' column must be numeric or categorial."):
        _check_prep_smd_data(
            sample_data,
            group='group',
            vars=['age', 'date']
        )

def test_check_prep_smd_data_missing_group(sample_data):
    with pytest.raises(ValueError, match="The DataFrame is missing the following required columns"):
        _check_prep_smd_data(
            sample_data,
            group='missing_group',
            vars=['age', 'race'],
            wt_var='ps_weights'
        )

def test_check_prep_smd_data_invalid_cat_vars(sample_data):
    with pytest.raises(ValueError, match="The DataFrame is missing the following required columns"):
        _check_prep_smd_data(
            sample_data,
            group='group',
            vars=['weight', 'race'],
            cat_vars=['invalid_cat']
        )

def test_check_prep_smd_data_missing_values(small_sample_data):
    data_with_nan = small_sample_data.copy()
    data_with_nan.loc[0, "binary_var"] = np.nan
    with pytest.raises(
        ValueError, match="The 'binary_var' column contains missing values."
    ):
        _check_prep_smd_data(
            data_with_nan, group="group", vars=["binary_var"], wt_var="weights"
        )

# Test --- _calc_smd_covar() -------------------------------------------------------

def test_calc_smd_covar_binary_unweighted(small_sample_data):
    smd = _calc_smd_covar(data=small_sample_data, group="group", covar="binary_var")
    assert isinstance(smd, float)
    assert smd > 0

def test_calc_smd_covar_binary_weighted(small_sample_data):
    smd = _calc_smd_covar(
        data=small_sample_data, group="group", covar="binary_var", wt_var="weights"
    )
    assert isinstance(smd, float)
    assert smd > 0

def test_calc_smd_covar_continuous_unweighted(small_sample_data):
    smd = _calc_smd_covar(data=small_sample_data, group="group", covar="cont_var")
    assert isinstance(smd, float)
    assert smd > 0

def test_calc_smd_covar_continuous_weighted(small_sample_data):
    smd = _calc_smd_covar(
        data=small_sample_data, group="group", covar="cont_var", wt_var="weights"
    )
    assert isinstance(smd, float)
    assert smd > 0


# --- zero variance and zero proportion ----------------------------------------------

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

def test_compute_smd_att_atc(small_sample_data):
    att_smd = _calc_smd_covar(
        data=small_sample_data,
        group="group",
        covar="cont_var",
        wt_var="weights",
        estimand="ATT",
    )
    assert isinstance(att_smd, float), "ATT SMD should return a float value"

    atc_smd = _calc_smd_covar(
        data=small_sample_data,
        group="group",
        covar="binary_var",
        wt_var="weights",
        estimand="ATC",
    )
    assert isinstance(atc_smd, float), "ATC SMD should return a float value"


# --- Testing compute_smd() --------------------------------------------------------

def test_compute_smd_invalid_group_type(sample_data):
    with pytest.raises(TypeError, match="The `group` parameter must be of type str"):
        compute_smd(sample_data, group=123, vars=["age"])

def test_compute_smd_invalid_vars_type_1(sample_data):
    with pytest.raises(TypeError, match="`vars` must be a list of strings"):
        compute_smd(sample_data, group="group", vars='age')

def test_compute_smd_invalid_vars_type_2(sample_data):
    with pytest.raises(TypeError, match="`vars` must be a list of strings"):
        compute_smd(sample_data, group="group", vars=1)

def test_compute_smd_invalid_cat_vars_type_1(sample_data):
    with pytest.raises(TypeError, match="`cat_vars` must be a list of strings"):
        compute_smd(sample_data, group="group", vars=['age'], cat_vars='race')

def test_compute_smd_invalid_cat_vars_type_2(sample_data):
    with pytest.raises(TypeError, match="`cat_vars` must be a list of strings"):
        compute_smd(sample_data, group="group", vars=['age'], cat_vars=2)        

def test_compute_smd_invalid_wt_var_type(sample_data):
    with pytest.raises(TypeError, match="`wt_var` parameter must be of type str"):
        compute_smd(sample_data, group="group", vars=['age'], wt_var=2)

def test_compute_smd_invalid_std_binary_type(sample_data):
    with pytest.raises(TypeError, match="`std_binary` parameter must be of type bool"):
        compute_smd(sample_data, group="group", vars=['age'], std_binary='yes')

