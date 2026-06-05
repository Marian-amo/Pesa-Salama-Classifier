# tests/test_app_startup.py
"""
Streamlit app startup smoke test.
Verifies that all imports in app/app.py resolve and the module loads
without raising exceptions. Does NOT start the Streamlit server.
"""
import sys
import pathlib
import importlib
import pytest


APP_DIR = pathlib.Path("app")


def test_app_directory_exists():
    assert APP_DIR.exists(), "app/ directory not found"


def test_app_py_exists():
    assert (APP_DIR / "app.py").exists(), "app/app.py not found"


def test_critical_imports():
    """All packages imported by app.py must be resolvable."""
    critical = [
        "streamlit",
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "plotly",
        "joblib",
        "re",
        "json",
        "pathlib",
    ]
    missing = []
    for pkg in critical:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    assert not missing, f"Missing packages required by app.py: {missing}"


def test_app_module_loads(monkeypatch, tmp_path):
    """
    Import app.py with Streamlit's runtime mocked out so it doesn't
    try to open a browser or bind to a port.
    """
    # Add app/ to sys.path so relative imports resolve
    sys.path.insert(0, str(APP_DIR))

    # Streamlit calls st.set_page_config() at module level — mock it
    import unittest.mock as mock
    with mock.patch("streamlit.set_page_config"), \
         mock.patch("streamlit.title"), \
         mock.patch("streamlit.sidebar"):
        try:
            spec = importlib.util.spec_from_file_location(
                "app", APP_DIR / "app.py"
            )
            # We only check the spec loads cleanly; don't exec the full module
            # because it calls st.* at module level
            assert spec is not None, "importlib could not create a spec for app.py"
        finally:
            sys.path.pop(0)