Make uploads possible: treat reading_id as stable and unique

Right now you do:

melted["reading_id"] = pd.RangeIndex(... start=1 ...)


That means:

reading_id restarts at 1 for every utility file

if someone concatenates all utilities later, reading_id is not unique

Better options:

Make a stable composite key (recommended):

reading_id = f"{utility}:{building_key}:{timestamp}" (string)

Or a stable integer hash:

reading_id = pd.util.hash_pandas_object([...])

Example (simple + stable):

melted["reading_id"] = (
    melted["utility"].astype(str) + ":" +
    melted["building_key"].astype(str) + ":" +
    melted["timestamp"].astype(str)
)

2) Huge speed improvement: chunked processing + streaming CSV write

Your current code reads an entire wide CSV into memory, melts it, then writes a huge CSV in one shot. That’s why it “takes forever” sometimes.

You can keep your logic but change to:

pd.read_csv(..., chunksize=...)

melt each chunk

append to output (mode="a")

keep a running counter if you still want integer IDs

This avoids massive memory spikes and makes long runs much more stable.

Pattern:

first = True
for chunk in pd.read_csv(csv_path, chunksize=50_000):
    melted = chunk.melt(...)
    ...
    melted.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
    first = False

3) Parse timestamps once (consistent + sortable)

Right now timestamp stays as a string. That can bite you later (sorting, filtering, timezone quirks).

Add:

melted["timestamp"] = pd.to_datetime(melted["timestamp"], errors="coerce", utc=True)


Then when writing to CSV you can write ISO:

melted["timestamp"] = melted["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")


Or keep datetime if you later move to Parquet.

4) Don’t melt weather/metadata, but validate them

You copy weather.csv and metadata.csv unchanged. Good.

But add lightweight checks so you fail fast if schema is off:

weather.csv must include timestamp

metadata.csv must include whatever building identifier you rely on (building_id etc.)

Example:

if name == "weather.csv":
    w = pd.read_csv(csv_path, nrows=1)
    assert "timestamp" in w.columns[0].lower()

5) Make metadata.csv building_key consistent with utility building_key

Right now you generate metadata building_key from building_id categories:

md["building_key"] = md["building_id"].astype("category").cat.codes + 1


That will not match your utility building_key mapping (which is built from column names in utility CSVs).

Better approach:

If metadata has a column that matches your utility building labels (often all_building, building, building_name, etc.), map using the SAME building_to_key.

Example:

for candidate in ["all_building", "building", "building_name"]:
    if candidate in md.columns:
        md["building_key"] = md[candidate].map(building_to_key).astype("Int64")
        break


If there’s no matching column, keep your current fallback.

6) Add CLI arguments (so the pipeline doesn’t require code edits)

Right now defaults are hard-coded. Add args:

--input-dir (default: drive_cache/original)

--output-dir (default: datasets/cleaned)

--chunksize (default: e.g. 50_000)

--drop-na (optional)

That makes it easy for collaborators + run_pipeline.py.

7) Optional: write Parquet for “big cleaned” (massive win)

If you ever decide Drive should store big cleaned data, CSV is the worst format.

Parquet is:

smaller

faster to write

faster to upload

faster to read later

Even if you keep CSV for cleaned_small, consider Parquet for cleaned.