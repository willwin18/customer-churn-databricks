"""
test_ingestion.py
Unit tests for data cleaning/ingestion logic.
"""

import pytest
from pyspark.sql import SparkSession

import sys
sys.path.append("../src")
from churn_pipeline.ingestion import (
    remove_cancelled_orders,
    remove_missing_customers,
    remove_invalid_rows,
)


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[1]").appName("test").getOrCreate()


def test_remove_cancelled_orders(spark):
    """Invoices starting with 'C' should be removed."""
    df = spark.createDataFrame(
        [("C1001",), ("1002",), ("C1003",)],
        ["Invoice"]
    )
    result = remove_cancelled_orders(df)
    invoices = [row["Invoice"] for row in result.collect()]
    assert invoices == ["1002"]


def test_remove_missing_customers(spark):
    """Rows with null Customer ID should be removed."""
    df = spark.createDataFrame(
        [(101,), (None,), (102,)],
        ["Customer ID"]
    )
    result = remove_missing_customers(df)
    ids = [row["Customer ID"] for row in result.collect()]
    assert None not in ids
    assert len(ids) == 2


def test_remove_invalid_rows_negative_quantity(spark):
    """Rows with negative or zero Quantity should be removed."""
    df = spark.createDataFrame(
        [(5, 10.0), (-2, 10.0), (0, 10.0)],
        ["Quantity", "Price"]
    )
    result = remove_invalid_rows(df)
    assert result.count() == 1


def test_remove_invalid_rows_negative_price(spark):
    """Rows with negative or zero Price should be removed."""
    df = spark.createDataFrame(
        [(5, 10.0), (5, -1.0), (5, 0.0)],
        ["Quantity", "Price"]
    )
    result = remove_invalid_rows(df)
    assert result.count() == 1
