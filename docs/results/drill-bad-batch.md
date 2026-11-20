# Drill report: bad-batch

- seed: 42
- weeks: 12
- versions: 1
- promotions: 0
- rejections: 0
- rollbacks: 0
- quarantined batches: 1

## Timeline

| tick | end | champion | WAPE-28d | numeric drift share | action |
| --- | --- | --- | --- | --- | --- |
| 1 | 2012-01-08 | v1 | 0.2109 | 0.75 | PROBATION_CONTINUE |
| 2 | 2012-01-15 | v1 | 0.2253 | 0.75 | PROBATION_CONTINUE |
| 3 | 2012-01-22 | v1 | 0.2291 | 0.75 | PROBATION_PASSED |
| 4 | 2012-01-29 | v1 | 0.2514 | 0.50 | SCORED |
| 5 | 2012-02-05 | v1 | 0.2572 | 0.50 | SCORED |
| 6 | 2012-02-12 | v1 | 0.2670 | 0.50 | SCORED |
| 7 | 2012-02-19 | v1 | 0.2743 | 0.50 | QUARANTINED |
| 8 | 2012-02-26 | v1 | 0.2473 | 0.75 | SCORED |
| 9 | 2012-03-04 | v1 | 0.2472 | 0.75 | SCORED |
| 10 | 2012-03-11 | v1 | 0.2424 | 1.00 | SCORED |
| 11 | 2012-03-18 | v1 | 0.2499 | 0.75 | SCORED |
| 12 | 2012-03-25 | v1 | 0.2838 | 0.50 | SCORED |

## Gate reports

No gate evaluation in this run.
## Summary

| metric | as-served | frozen v1 |
| --- | --- | --- |
| RMSE | 70.3356 | 70.3356 |
| MAE | 43.4195 | 43.4195 |
| WAPE | 0.2654 | 0.2654 |
| rows | 1840 | 1840 |

as-served RMSE is 1.00x the frozen-v1 RMSE.

promotions=0 rejections=0 rollbacks=0 quarantined=1
