import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="session")
def _run_from_repo_root():
    """Config paths (app/config/routes.yaml, .env) are relative, so tests run
    from the repo root regardless of where pytest was invoked."""
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(previous)
