# Databricks notebook source
import pytest
import sys

# May not have write permissions on databricks, so prevent __pycache__ creation
sys.dont_write_bytecode = True


def run_pytest_main(flags: list[str]):
    """
    Run pytest with the given flags.

    Example: run_pytest_main([".", "-v", "-p", "no:cacheprovider"])
    """
    retcode = pytest.main(flags)
    assert retcode == 0, "The pytest invocation failed. See the log for details"


# COMMAND ----------

# MAGIC %md
# MAGIC Pytest call meanings
# MAGIC
# MAGIC | flag | definition |
# MAGIC |------|------------|
# MAGIC | "." | Run all tests in cwd, this translates to tsdb/tests |
# MAGIC | -v | Verbose |
# MAGIC | -p no:cacheprovider | -p specifies plugins, this full command prevents pytest from caching results (since we can't write) |
# MAGIC
# MAGIC This is a basic pytest call and you can have more complex setups by passing different flags.

# COMMAND ----------

run_pytest_main(["test_datasets.py", "-v", "-p", "no:cacheprovider"])
