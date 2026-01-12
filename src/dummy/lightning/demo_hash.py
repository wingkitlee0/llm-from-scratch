from ray.data.expressions import udf, col
from ray.data.datatype import DataType
import pyarrow as pa
import ray
import duckdb


@udf(return_dtype=DataType.uint64)
def hash_udf(x: pa.ChunkedArray, y: pa.ChunkedArray) -> pa.ChunkedArray:
    arrow_table = pa.Table.from_pydict({"x": x, "y": y})

    # Concatenate with separator, THEN hash (not hash then XOR)
    # The separator '|' prevents collisions like concat('1','23') vs concat('12','3')
    rel = duckdb.sql("""
        SELECT md5_number_lower(x::VARCHAR || '|' || y::VARCHAR) as h
        FROM arrow_table
    """)

    res = rel.arrow()
    if isinstance(res, pa.RecordBatchReader):
        res = res.read_all()
    return res.column("h")

if __name__ == "__main__":
    ds = (
        ray.data.range(100_000_000)
        .with_column("x", col("id"))
        .with_column("y", col("id") * 2)
        .with_column("hash", hash_udf(col("x"), col("y")))
    )

    ds.repartition(10).write_parquet("hash_udf.parquet", mode="overwrite")

    # Rewrite
    (
        ray.data.read_parquet("hash_udf.parquet")
        .sort("hash")
        .write_parquet("hash_udf_sorted.parquet", mode="overwrite")
    )

