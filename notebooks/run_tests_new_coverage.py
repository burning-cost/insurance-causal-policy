# Databricks notebook source
# MAGIC %pip install insurance-causal-policy pytest

# COMMAND ----------

# Install the package and clone the repo to get new tests
import subprocess
import sys
import os

# Clone or update the repo
result = subprocess.run(
    ["git", "clone", "https://github.com/burning-cost/insurance-causal-policy.git",
     "/tmp/insurance-causal-policy-new"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# COMMAND ----------

# Install in editable mode so we pick up latest source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/insurance-causal-policy-new"],
    capture_output=True, text=True
)
print(result.stdout[-3000:])
if result.stderr:
    print("STDERR:", result.stderr[-1000:])

# COMMAND ----------

# Run only the new coverage tests first
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_new_coverage.py",
     "--tb=short", "-v", "--no-header", "-q"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-causal-policy-new"
)
print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)

# COMMAND ----------

# Run the full test suite
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--tb=short", "-v", "--no-header", "-q",
     "--ignore=tests/test_new_coverage.py"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-causal-policy-new"
)
print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
