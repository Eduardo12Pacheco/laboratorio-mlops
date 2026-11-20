# Drill report: rollback

- seed: 42
- weeks: 25
- versions: 3
- promotions: 2
- rejections: 0
- rollbacks: 1
- quarantined batches: 0

## Timeline

| tick | end | champion | WAPE-28d | numeric drift share | action |
| --- | --- | --- | --- | --- | --- |
| 1 | 2012-01-08 | v1 | 0.2109 | 0.75 | PROBATION_CONTINUE |
| 2 | 2012-01-15 | v1 | 0.2253 | 0.75 | PROBATION_CONTINUE |
| 3 | 2012-01-22 | v1 | 0.2291 | 0.75 | PROBATION_PASSED |
| 4 | 2012-01-29 | v1 | 0.2514 | 0.50 | SCORED |
| 5 | 2012-02-05 | v1 | 0.2572 | 0.50 | SCORED |
| 6 | 2012-02-12 | v1 | 0.2670 | 0.50 | SCORED |
| 7 | 2012-02-19 | v1 | 0.2715 | 0.50 | SCORED |
| 8 | 2012-02-26 | v1 | 0.2511 | 0.75 | SCORED |
| 9 | 2012-03-04 | v1 | 0.2510 | 0.50 | SCORED |
| 10 | 2012-03-11 | v1 | 0.2471 | 0.75 | SCORED |
| 11 | 2012-03-18 | v1 | 0.2499 | 0.75 | SCORED |
| 12 | 2012-03-25 | v1 | 0.2838 | 0.50 | SCORED |
| 13 | 2012-04-01 | v2 | 0.3086 | 0.50 | PROMOTED |
| 14 | 2012-04-08 | v2 | 0.1097 | 0.75 | PROBATION_CONTINUE |
| 15 | 2012-04-15 | v2 | 0.1220 | 0.75 | PROBATION_CONTINUE |
| 16 | 2012-04-22 | v2 | 0.1420 | 0.75 | PROBATION_PASSED |
| 17 | 2012-04-29 | v2 | 0.1469 | 0.75 | SCORED |
| 18 | 2012-05-06 | v2 | 0.1594 | 0.75 | SCORED |
| 19 | 2012-05-13 | v2 | 0.1570 | 0.75 | SCORED |
| 20 | 2012-05-20 | v2 | 0.1517 | 0.50 | SCORED |
| 21 | 2012-05-27 | v2 | 0.1457 | 0.50 | SCORED |
| 22 | 2012-06-03 | v2 | 0.1449 | 0.50 | SCORED |
| 23 | 2012-06-10 | v3 | 0.1397 | 0.50 | PROMOTION_FORCED |
| 24 | 2012-06-17 | v2 | 0.7240 | 0.75 | ROLLBACK |
| 25 | 2012-06-24 | v2 | 0.1375 | 0.75 | SCORED |

## Gate reports

### tick 13 - v2 - APPROVED

| gate | passed | observed | threshold |
| --- | --- | --- | --- |
| DATA_VALID | True | WARN | not FAIL |
| ABS_WAPE | True | 0.1470 | 0.3000 |
| BEATS_BASELINE | True | 53.2673 | 126.6705 |
| IMPROVEMENT | True | 0.5595 | 0.0500 |
| SIGNIFICANCE | True | 9000.8429 | 0.0000 |
| SEGMENT | True | -0.5528 | 0.1000 |

### tick 23 - v3 - REJECTED

| gate | passed | observed | threshold |
| --- | --- | --- | --- |
| DATA_VALID | True | WARN | not FAIL |
| ABS_WAPE | False | 0.7200 | 0.3000 |
| BEATS_BASELINE | False | 267.9086 | 150.8285 |
| IMPROVEMENT | False | -4.0620 | 0.0500 |
| SIGNIFICANCE | False | -77946.8003 | 0.0000 |
| SEGMENT | False | 4.6943 | 0.1000 |

## Summary

| metric | as-served | frozen v1 |
| --- | --- | --- |
| RMSE | 84.8111 | 88.8369 |
| MAE | 48.6398 | 58.7777 |
| WAPE | 0.2287 | 0.2763 |
| rows | 4190 | 4190 |

as-served RMSE is 0.95x the frozen-v1 RMSE.

promotions=2 rejections=0 rollbacks=1 quarantined=0
