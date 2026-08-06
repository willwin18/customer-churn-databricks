"""
ingestion.py
Handles loading and initial cleaning of raw retail transaction data.
"""

from pyspark.sql import DataFrame, SparkSession


def load_raw_transactions(spark: SparkSession, table_name: str = "online_retail_ii") -> DataFrame:
    """Load the raw transactions table."""
    return spark.table(table_name)


def remove_cancelled_orders(df: DataFrame) -> DataFrame:
    """Remove rows where Invoice starts with 'C' (cancellations)."""
    return df.filter(~df["Invoice"].startswith("C"))


def remove_missing_customers(df: DataFrame) -> DataFrame:
    """Remove rows with no Customer ID."""
    return df.filter(df["Customer ID"].isNotNull())


def remove_invalid_rows(df: DataFrame) -> DataFrame:
    """Remove rows with non-positive quantity or price."""
    return df.filter((df["Quantity"] > 0) & (df["Price"] > 0))


def clean_transactions(spark: SparkSession) -> DataFrame:
    """Full cleaning pipeline: load, remove cancellations, nulls, invalid rows."""
    df = load_raw_transactions(spark)
    df = remove_cancelled_orders(df)
    df = remove_missing_customers(df)
    df = remove_invalid_rows(df)
    return df
