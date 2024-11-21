# Databricks notebook source
# MAGIC %md
# MAGIC # Get and Set Configurations
# MAGIC
# MAGIC The purpose of this notebook is to get the tags from cluster, get the catalog and schema values from the information schema and read configuratons values from the config file. Create a global temperary view, these values are fetched in various notebooks.
# MAGIC
# MAGIC ## Inputs
# MAGIC * Cluster Tags
# MAGIC * Information Schema
# MAGIC * configs/cf_towerscout_config.json
# MAGIC
# MAGIC ## Processes
# MAGIC * Read cluster tags 1. Environment, 2. Configuration file name, project name.
# MAGIC * Get Catalog and Schema Name based on the env
# MAGIC * Read the config json file
# MAGIC * Set the values in variables
# MAGIC * Create global temp view
# MAGIC
# MAGIC ## Outputs
# MAGIC * Global temp view created with all the configuration values to be accessed by other notebooks.

# COMMAND ----------

import json

# COMMAND ----------

def get_cluster_tag_val(key_name: str) -> str:
    # Fetch all cluster tags from the Spark configuration
    all_tags = spark.conf.get("spark.databricks.clusterUsageTags.clusterAllTags", None)

    if all_tags:
        # Parse the JSON-like string into a list of dictionaries
        tags_list = json.loads(all_tags)

        # Convert the list of dictionaries into a dictionary for easier access
        tags_dict = {tag["key"]: tag["value"] for tag in tags_list}

        # Get the value of the specified tag
        tag_value = tags_dict.get(key_name)

        if tag_value:
            # Return the tag value if it exists
            return tag_value
        else:
            # Exit the notebook with a failure message if the tag does not exist or has no value
            dbutils.notebook.exit(
                f"Failure: The tag '{key_name}' does not exist or has no value."
            )
    else:
        # Display a message if no tags are found or the cluster does not have any tags
        display("No tags found or the cluster does not have any tags.")
        dbutils.notebook.exit(
            f"Failure: The tag '{key_name}' does not exist or has no value."
        )


# This function retrieves the catalog name, schema name, and volumes using the filtered catalog and schema names
def get_catalog_schema_config(cat, schema):
    # Query to get the catalog name and schema name from the information schema
    conf = spark.sql(
        f"SELECT catalog_name, schema_name FROM information_schema.schemata WHERE catalog_name LIKE '%{cat}%'"
    )
    # Extract the catalog name and schema name from the query result
    catalog_name = conf.collect()[0]["catalog_name"]
    schema_name = "towerscout"

    # Query to get the volumes in the specified catalog and schema
    vols = spark.sql(f"show volumes in {catalog_name}.{schema_name}")

    # Extract the volume names from the query result
    volumes = [row["volume_name"] for row in vols.select("volume_name").collect()]
    return catalog_name, schema_name, volumes

# COMMAND ----------

env = get_cluster_tag_val("edav_environment")

edav_center = get_cluster_tag_val(
    "edav_center"
).lower()  # must lowercase or else LIKE statement will not work

edav_project = get_cluster_tag_val("edav_project").lower()

cluster_catalog_name, cluster_schema_name, volumes = get_catalog_schema_config(
    edav_center, edav_project
)

# Retrieve the config file path from cluster tags
cluster_config_path = get_cluster_tag_val(
    "tower_scout_config_file_name"
).strip()  # for some reason there is a leading space

# Set the retrieved environment, catalog name, and schema name as Spark configuration variables
spark.conf.set("env", env)
spark.conf.set("catalog_name", cluster_catalog_name)
spark.conf.set("schema_name", cluster_schema_name)
spark.conf.set("config_path", cluster_config_path)

if "configs" in volumes:
    # Construct the volume location path for the config file
    vol_loc = f"/Volumes/{cluster_catalog_name}/{cluster_schema_name}/configs/{cluster_config_path}"
    try:
        # Load the configuration data from the specified volume location
        conf = spark.read.format("json").load(vol_loc, multiLine=True)
        tenant_id = conf.collect()[0]["tenant_id"]
        # Set the debug mode from the loaded configuration data based on the environment
        if env == "development" or env == "production":
            spark.conf.set("debug_mode", conf.collect()[0][env]["debug_mode"])
            spark.conf.set("unit_test_mode", conf.collect()[0][env]["unit_test_mode"])
            spark.conf.set("key_vault_name", conf.collect()[0][env]["key_vault_name"])
            spark.conf.set("batch_size", conf.collect()[0][env]["batch_size"])
            spark.conf.set("bronze_path", conf.collect()[0][env]["bronze_path"])
            spark.conf.set("silver_table_name", conf.collect()[0][env]["silver_table_name"])
            client_id = conf.collect()[0][env]["azure_config"]["client_id"]
            writestream_trigger_args = conf.collect()[0][env]["writestream_trigger_args"]
            image_config = conf.collect()[0][env]["image_config"]
        else:
            dbutils.notebook.exit(
                f"Failure: Enviornment is neither development or production is {env}."
            )
    except Exception as e:
        # Exit the notebook with an error message if unable to load the configuration data
        dbutils.notebook.exit(
            f"Failure: Unable to load configuration data, check config file. Error: {str(e)}"
        )
else:
    # Exit the notebook with an error message if the 'configs' volume does not exist
    dbutils.notebook.exit("No config folder in volume")

spark.conf.set(
    "vol_location_configs",
    f"/Volumes/{cluster_catalog_name}/{spark.conf.get('schema_name')}/configs/",
)

# COMMAND ----------

config_data = [
    (
        spark.conf.get("env"),
        spark.conf.get("catalog_name"),
        spark.conf.get("schema_name"),
        spark.conf.get("debug_mode"),
        spark.conf.get("vol_location_configs"),
        spark.conf.get("key_vault_name"),
        spark.conf.get("silver_table_name"),
        spark.conf.get("batch_size"),
        spark.conf.get("bronze_path"),
        image_config,
        writestream_trigger_args,
        client_id,
        tenant_id,
        spark.conf.get("unit_test_mode"),
    )
]

config_columns = [
    "env",
    "catalog_name",
    "schema_name",
    "debug_mode",
    "vol_location_configs",
    "key_vault_name",
    "silver_table_name",
    "batch_size",
    "bronze_path",
    "image_config",
    "writestream_trigger_args",
    "client_id",
    "tenant_id",
    "unit_test_mode",
]

# Create a DataFrame with the configuration values and column names
config_df = spark.createDataFrame(config_data, config_columns)

# Create a global temporary view for the configuration DataFrame
config_df.createOrReplaceGlobalTempView("global_temp_towerscout_configs")

display(spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs"))
