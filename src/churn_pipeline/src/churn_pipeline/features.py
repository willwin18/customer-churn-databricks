"""
features.py
Builds customer-level features from cleaned transaction data.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum as _sum, countDistinct, max as _max, datediff, lit, when


def add_total_price(df: DataFrame) -> DataFrame:
    """Add TotalPrice column (Quantity * Price)."""
    return df.withColumn("TotalPrice", col("Quantity") * col("Price"))


def build_customer_summary(df: DataFrame) -> DataFrame:
    """Aggregate transaction-level data into one row per customer."""
    return df.groupBy("Customer ID").agg(
        _sum("TotalPrice").alias("total_spend"),
        countDistinct("Invoice").alias("num_orders"),
        _max("InvoiceDate").alias("last_purchase_date")
    )


def add_recency(df: DataFrame, reference_date: str) -> DataFrame:
    """Add days_since_last_purchase relative to a reference date."""
    return df.withColumn(
        "days_since_last_purchase",
        datediff(lit(reference_date), col("last_purchase_date"))
    )


def add_churn_label(df: DataFrame, churn_threshold_days: int = 90) -> DataFrame:
    """Label customers as churned (1) if inactive beyond the threshold."""
    return df.withColumn(
        "churned",
        when(col("days_since_last_purchase") > churn_threshold_days, 1).otherwise(0)
    )


def build_customer_features(df: DataFrame, reference_date: str, churn_threshold_days: int = 90) -> DataFrame:
    """Full feature engineering pipeline."""
    df = add_total_price(df)
    customer_df = build_customer_summary(df)
    customer_df = add_recency(customer_df, reference_date)
    customer_df = add_churn_label(customer_df, churn_threshold_days)
    return customer_df.withColumnRenamed("Customer ID", "CustomerID")
