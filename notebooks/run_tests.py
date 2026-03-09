# Databricks notebook source
# MAGIC %pip install insurance-causal-policy pytest

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--tb=short", "-v", "--no-header", "-q"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-causal-policy"
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
