# -*- coding: utf-8 -*-
"""
Section 4 - PySpark Application on Hadoop
-----------------------------------------
Reads the cleaned penguin dataset from HDFS, shows its schema, selects the
relevant columns, removes invalid flipper lengths (<= 0), sorts, caches and
writes the result to HDFS as JSON partitioned by species.

Run on the HDP sandbox:
    spark-submit --master yarn --deploy-mode client spark/penguins_spark.py
(or --master local[*] for a quick test)

Compatible with Spark 2.x / 3.x and Python 2.7 / 3.x (no f-strings).
"""
from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_PATH = "hdfs:///user/maria_dev/datasets/clean/penguins_clean.csv"
OUTPUT_PATH = "hdfs:///user/maria_dev/spark/output/penguins"


def main():
    spark = (SparkSession.builder
             .appName("PenguinsSparkApp")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        # 4.1 Read cleaned CSV from HDFS and display schema
        df = (spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(INPUT_PATH))

        print("=== Schema of penguins_clean.csv ===")
        df.printSchema()
        print("Total input records: {0}".format(df.count()))
        df.show(5, truncate=False)

        # 4.2 Select relevant columns
        selected = df.select(
            "species", "island", "sex", "year",
            "bill_length_mm", "bill_depth_mm",
            "flipper_length_mm", "body_mass_kg"
        )

        # Remove records with flipper length <= 0 (invalid measurements)
        invalid = selected.filter(F.col("flipper_length_mm") <= 0).count()
        print("Records with flipper_length_mm <= 0 (removed): {0}".format(invalid))
        valid = selected.filter(F.col("flipper_length_mm") > 0)

        # Simple filter + sorting: heavier penguins first, then longest flippers
        result = (valid
                  .filter(F.col("body_mass_kg") >= 3.0)
                  .orderBy(F.col("body_mass_kg").desc(),
                           F.col("flipper_length_mm").desc()))

        # Cache results because they are reused (count, show, write)
        result.cache()
        print("Records after filtering & sorting (cached): {0}".format(result.count()))
        result.show(10, truncate=False)

        print("=== Summary per species ===")
        (result.groupBy("species")
               .agg(F.count("*").alias("n"),
                    F.round(F.avg("body_mass_kg"), 3).alias("avg_body_mass_kg"),
                    F.round(F.avg("flipper_length_mm"), 2).alias("avg_flipper_mm"))
               .orderBy("species")
               .show())

        # 4.3 Save as JSON, partitioned by species
        # repartition by species -> one compact file per partition folder
        (result.repartition("species")
               .sortWithinPartitions(F.col("body_mass_kg").desc())
               .write
               .mode("overwrite")
               .partitionBy("species")
               .json(OUTPUT_PATH))
        print("Output written to {0} (partitioned by species)".format(OUTPUT_PATH))

        result.unpersist()
    finally:
        # Stop the Spark session properly
        spark.stop()
        print("Spark session stopped.")


if __name__ == "__main__":
    main()
