# Databricks notebook source
# MAGIC %pip install pytest

# COMMAND ----------

import pytest
import sys

# COMMAND ----------

sys.dont_write_bytecode = True
retcode = pytest.main([".", "-v", "-p", "no:cacheprovider"])

assert retcode == 0, "The pytest invocation failed. See the log for details"

# COMMAND ----------


