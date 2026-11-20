# Data

One problem, one dataset: hourly bike rentals in Washington DC, 2011–2012, from the UCI
Machine Learning Repository. The dataset is bundled in the wheel so the project works
fully offline; the optional acquisition command exists only to reproduce the bundle.

## Provenance

| Field | Value |
|---|---|
| Dataset | Bike Sharing Dataset (hourly) |
| Author | Hadi Fanaee-T (2013) |
| Landing page | <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset> |
| DOI | [10.24432/C5W894](https://doi.org/10.24432/C5W894) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Pinned archive | <https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip> |
| Bundled file | `src/mlops_lifecycle_lab/_data/hour.csv.gz` |
| Manifest | `src/mlops_lifecycle_lab/_data/manifest.json` |
| Attribution | `src/mlops_lifecycle_lab/_data/ATTRIBUTION.txt` |

The bundled `.csv.gz` is an unmodified gzip compression of `hour.csv` from the pinned
archive. The decompressed bytes are the identity that matters and are what the CLI
verifies.

## Checksums

| Artifact | sha256 | Size |
|---|---|---|
| UCI zip | `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401` | 279,992 B |
| `hour.csv` (decompressed) | `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f` | 1,156,736 B |
| `hour.csv.gz` (committed) | derived by gzip; not a provenance identity | 256,659 B |

`day.csv` (731 rows) and `Readme.txt` are present in the archive but unused. The archive
`Readme.txt` is the source for the normalisation constants (`temp` uses −8…+39 °C,
`atemp` uses −16…+50 °C, `hum` and `windspeed` are divided by 100 and 67 respectively).

## Schema

Raw contract (`mll validate`, batch source, serving API), in order:

```text
entity_id, timestamp, season, holiday, workingday, weathersit,
temp, atemp, hum, windspeed
```

Labels (archive and training only): `casual, registered, cnt`; the target is `cnt`, and
`cnt == casual + registered` on every row. `timestamp` is derived from the UCI
`dteday` + `hr` columns and `entity_id` is the UCI `instant`. The feature contract
(`FEATURE_COLUMNS`) is defined once in `domain/schema.py`; `yr`, `instant`, `casual`,
`registered` and `cnt` are leakage/time proxies and never become features.

## Lineage

```text
UCI archive zip (pinned sha256)
  └─ hour.csv (decompressed sha256)
       └─ _data/hour.csv.gz + _data/manifest.json (committed, shipped in the wheel)
            ├─ mll data verify            -> provenance check (exit 0/2)
            ├─ mll data info --json       -> rows, window, checksum, license
            ├─ BundledDataset.raw()       -> raw schema -> validation / features
            ├─ BundledDataset.archive()   -> 2011 rows for the bootstrap champion
            └─ BundledDataset.window()    -> 2012 weekly replay batches
```

`mll data fetch --source uci --dest DIR` re-downloads the pinned zip, verifies the zip
sha256 and then the inner `hour.csv` sha256 before writing anything. It is optional,
never used by tests or the default runtime path, and refuses mismatches with
`ChecksumMismatchError` (CLI exit code 2).

## Validation rules

`mll validate` implements the data contract in `domain/validation.py`. It is a pure
function: rules are all evaluated (no short-circuit) and the input frame is never
mutated. A failed `ERROR` rule makes the verdict `FAIL`; a failed `WARN` rule makes it
`WARN`; otherwise `PASS`. A `FAIL` batch is quarantined by the lifecycle tick: never
scored, never logged, never trained on.

| Rule | Context | Severity | Tolerance |
|---|---|---|---|
| `required_columns` | both | ERROR | 0 |
| `no_nulls` | both | ERROR | 0 |
| `timestamp_hourly` | both | ERROR | 0 |
| `timestamp_unique` | both | ERROR | 0 |
| `timestamp_monotonic` | both | ERROR | 0 |
| `season_domain` (1–4) | both | ERROR | 0 |
| `binary_domain` (`holiday`, `workingday`) | both | ERROR | 0 |
| `weathersit_domain` (1–4) | both | ERROR | 0 |
| `numeric_range` (`temp/atemp/hum/windspeed` ∈ [0,1]) | both | ERROR | 0 |
| `hum_positive` | both | WARN | 0 |
| `gap_hours` (max gap ≤ 48 h) | both | WARN | 0 |
| `label_consistency` (`cnt == casual + registered`, `cnt ≥ 0`) | training | ERROR | 0 |
| `batch_size` (1…1000 rows) | batch | ERROR | 0 |

The bundled dataset validates as `WARN` with exactly one failed rule: `hum_positive`,
22 rows, all on 2011-03-10. There are zero failed `ERROR` rules. The plan mentioned a
0.5% tolerance for `hum_positive`; S3 and the T2 acceptance require those 22 rows to be
reported as the single warning, so the implemented tolerance is zero and any zero
humidity is visible.

## Known quirks

- The UCI landing page says 17,389 instances; the actual `hour.csv` has 17,379 rows.
- 165 hours are absent because there were no rentals in those hours (0 nulls, 0
  duplicate timestamps). The largest gap between consecutive rows is 1 day 13 hours.
- `windspeed = 0` on 2,180 rows; `weathersit = 4` on only 3 rows.
- `hum = 0` on 22 rows, all on 2011-03-10, which is why the dataset is `WARN`, not
  `PASS`.
- The committed gzip is produced with `gzip -9 -n` (no filename or mtime in the
  header), hence 256,659 bytes; the planning note's 256,668 bytes came from a gzip run
  that stored the header metadata. The decompressed sha256 is the binding identity.
- The replay treats the 2012 half as production. Labels arrive one tick late by design;
  the data itself has no missing labels.

## Result artifacts

The committed drill results are generated by:

```bash
uv run mll drill run --scenario all \
  --report-dir docs/results --workspace "$(mktemp -d)"
uv run python scripts/render_results_table.py
```

`docs/results/drill-*.json` is the reproducible source of truth (`schema:
drill-report/v1`, seed 42); the `.md` files are rendered from it and
[`results-table.md`](results/results-table.md) is generated from all four JSON reports by
`scripts/render_results_table.py --check` in the docs gate. The two curated Evidently
reports under `assets/reports/` use the dataset windows described there and share one
extracted JavaScript bundle so the directory stays under 5 MB; regenerate them with
`uv run python scripts/render_drift_reports.py`.

## Reproducing the bundle

```bash
tmp="$(mktemp -d)"
uv run mll data fetch --source uci --dest "$tmp"
gzip -9 -n -c "$tmp/hour.csv" > "$tmp/hour.csv.gz"
uv run mll data verify --path "$tmp/hour.csv.gz"
uv run mll validate --bundled --json
```
