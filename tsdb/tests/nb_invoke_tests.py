# Databricks notebook source
# MAGIC %pip install pytest

# COMMAND ----------

import pytest
import sys

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

# May not have write permissions on databricks, so prevent __pycache__ creation
sys.dont_write_bytecode = True
retcode = pytest.main([".", "-v", "-p", "no:cacheprovider"])

assert retcode == 0, "The pytest invocation failed. See the log for details"

# COMMAND ----------


