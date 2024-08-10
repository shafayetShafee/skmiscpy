import os
import sys
import pytest
from pathlib import Path
from skmiscpy import here
from skmiscpy.here import _get_project_root


@pytest.fixture(autouse=True)
def clear_cache():
    _get_project_root.cache_clear()
    yield
    _get_project_root.cache_clear()


@pytest.fixture
def mock_venv(monkeypatch, tmp_path):
    """Fixture to mock a virtual environment directory and return the project root directory."""
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    project_root = venv_path.parent
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_path))
    monkeypatch.setattr(sys, "prefix", str(venv_path))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
    return project_root


def test_here_valid_path(mock_venv):
    test_file = mock_venv / "test_folder" / "test_file.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("test")
    result = here("test_folder/test_file.txt")
    expected = str(mock_venv / "test_folder" / "test_file.txt")
    assert result == expected


def test_here_empty_path(mock_venv):
    with pytest.raises(
        ValueError, match="The `path` parameter cannot be an empty string."
    ):
        here("")


def test_here_non_string_path(mock_venv):
    with pytest.raises(TypeError, match="The `path` parameter must be a string."):
        here(123)


def test_here_absolute_path(mock_venv):
    with pytest.raises(
        ValueError, match="The `path` parameter must be relative, not absolute."
    ):
        here("/absolute/path")


def test_here_no_venv(monkeypatch):
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    with pytest.raises(OSError, match="Virtual environment is not activated."):
        here("test_path")


def test_here_no_virtual_env_var(mock_venv, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV")
    with pytest.raises(
        OSError, match="The VIRTUAL_ENV environment variable is not set or is empty."
    ):
        here("test_path")


def test_here_non_existent_venv(mock_venv, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/non/existent/path")
    with pytest.raises(
        OSError, match="The directory specified by VIRTUAL_ENV .* does not exist."
    ):
        here("test_path")


def test_here_parent_directory(mock_venv):
    (mock_venv / "parent_folder").mkdir()
    result = here("..")
    expected = str(mock_venv.parent)
    assert result == expected


def test_here_current_directory(mock_venv):
    result = here(".")
    assert os.path.samefile(result, str(mock_venv))


def test_here_nested_path(mock_venv):
    nested_folder = mock_venv / "folder1" / "folder2"
    nested_folder.mkdir(parents=True, exist_ok=True)
    result = here("folder1/folder2/file.txt")
    expected = str(mock_venv / "folder1" / "folder2" / "file.txt")
    assert result == expected


def test_here_with_env_var(mock_venv, monkeypatch):
    monkeypatch.setenv("TEST_ENV_VAR", "test_value")
    result = here("$TEST_ENV_VAR/file.txt")
    expected = str(mock_venv / "test_value" / "file.txt")
    assert result == expected


def test_here_with_spaces(mock_venv):
    (mock_venv / "folder with spaces").mkdir()
    result = here("folder with spaces/file with spaces.txt")
    expected = str(mock_venv / "folder with spaces" / "file with spaces.txt")
    assert result == expected


def test_here_with_unicode(mock_venv):
    (mock_venv / "folder_áéíóú").mkdir()
    result = here("folder_áéíóú/file_ñ.txt")
    expected = str(mock_venv / "folder_áéíóú" / "file_ñ.txt")
    assert result == expected


def test_here_cache(mock_venv):
    _get_project_root.cache_clear()

    here("test_file.txt")
    here("test_file.txt")

    assert _get_project_root.cache_info().hits == 1
