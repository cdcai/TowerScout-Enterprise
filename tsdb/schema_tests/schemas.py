from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, ArrayType

# Define the statistics_schema
statistics_schema = StructType([
    StructField("mean", ArrayType(FloatType())),
    StructField("median", ArrayType(IntegerType())),
    StructField("stddev", ArrayType(FloatType())),
    StructField("extrema", ArrayType(ArrayType(IntegerType()))),
])
