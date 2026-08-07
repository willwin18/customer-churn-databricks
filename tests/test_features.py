"""
test_features.py
Unit tests for feature engineering logic.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import sys
sys.path.append("../src")
from churn_pipeline.features import (
    add_total_price,
    build_customer_summary,
    add_recency,
    add_churn_label,
)


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[1]").appName("test").getOrCreate()


def test_add_total_price(spark):
    """TotalPrice should equal Quantity * Price."""
    df = spark.createDataFrame(
        [(2, 3.25), (1, 4.00)],
        ["Quantity", "Price"]
    )
    result = add_total_price(df)
    values = [row["TotalPrice"] for row in result.select("TotalPrice").collect()]
    assert values == [6.50, 4.00]


def test_add_total_price_zero_quantity(spark):
    """TotalPrice should be 0 if Quantity is 0."""
    df = spark.createDataFrame([(0, 10.0)], ["Quantity", "Price"])
    result = add_total_price(df)
    value = result.collect()[0]["TotalPrice"]
    assert value == 0.0


def test_build_customer_summary_aggregates_correctly(spark):
    """Customer summary should correctly sum spend and count distinct orders."""
    df = spark.createDataFrame(
        [
            ("C001", "INV1", 10.0, "2024-01-01"),
            ("C001", "INV1", 20.0, "2024-01-01"),
            ("C001", "INV2", 15.0, "2024-02-01"),
        ],
        ["Customer ID", "Invoice", "TotalPrice", "InvoiceDate"]
    )
    df = df.withColumn("InvoiceDate", col("InvoiceDate").cast("timestamp"))
    result = build_customer_summary(df).collect()[0]

    assert result["total_spend"] == 45.0
    assert result["num_orders"] == 2


def test_add_recency_calculates_days_correctly(spark):
    """days_since_last_purchase should be correct given a reference date."""
    df = spark.createDataFrame(
        [("2024-01-01",)], ["last_purchase_date"]
    ).withColumn("last_purchase_date", col("last_purchase_date").cast("timestamp"))

    result = add_recency(df, reference_date="2024-01-31")
    value = result.collect()[0]["days_since_last_purchase"]
    assert value == 30


def test_add_churn_label_flags_correctly(spark):
    """Customers over the threshold should be labeled churned=1, others 0."""
    df = spark.createDataFrame(
        [(100,), (50,), (91,), (90,)],
        ["days_since_last_purchase"]
    )
    result = add_churn_label(df, churn_threshold_days=90).collect()

    labels = {row["days_since_last_purchase"]: row["churned"] for row in result}
    assert labels[100] == 1
    assert labels[50] == 0
    assert labels[91] == 1
    assert labels[90] == 0


def test_churned_column_only_has_0_or_1(spark):
    """churned column should only ever contain 0 or 1, no other values."""
    df = spark.createDataFrame(
        [(10,), (200,), (89,)],
        ["days_since_last_purchase"]
    )
    result = add_churn_label(df).collect()
    values = set(row["churned"] for row in result)
    assert values.issubset({0, 1})
