"""Pytest configuration for insurance-causal-policy tests."""
import warnings
import pytest

# Suppress CVXPY solver verbose output during tests
import os
os.environ.setdefault("CVXPY_CACHE_SIZE", "0")
