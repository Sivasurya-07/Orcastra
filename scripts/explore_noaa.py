"""Explore NOAA GeoParquet schema and test spatial query."""
import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("INSTALL spatial; LOAD spatial;")

url = "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ais-2024-01-15.parquet"

# Get full schema
schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}') LIMIT 0").fetchall()
print("FULL SCHEMA:")
for row in schema:
    print(f"  {row[0]}: {row[1]}")

# Try reading with spatial extraction
try:
    result = con.execute(f"""
        SELECT mmsi, base_date_time, sog, cog, heading, vessel_name, vessel_type, status,
               ST_Y(ST_GeomFromWKB(geometry)) as lat,
               ST_X(ST_GeomFromWKB(geometry)) as lon
        FROM read_parquet('{url}')
        LIMIT 5
    """).fetchdf()
    print("\nSAMPLE (ST_GeomFromWKB):")
    print(result.to_string())
except Exception as e:
    print(f"\nST_GeomFromWKB failed: {e}")
    # Try without WKB conversion
    try:
        result = con.execute(f"""
            SELECT mmsi, base_date_time, sog, cog, heading, vessel_type,
                   ST_Y(geometry) as lat, ST_X(geometry) as lon
            FROM read_parquet('{url}')
            LIMIT 5
        """).fetchdf()
        print("\nSAMPLE (direct ST_Y/ST_X):")
        print(result.to_string())
    except Exception as e2:
        print(f"\nDirect ST_Y/ST_X failed: {e2}")
        # Just print geometry column type/sample
        result = con.execute(f"""
            SELECT typeof(geometry), geometry
            FROM read_parquet('{url}')
            LIMIT 3
        """).fetchdf()
        print("\nGeometry type and samples:")
        print(result.to_string())

con.close()
