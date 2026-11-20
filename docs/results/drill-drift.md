# Drill report: drift

- seed: 42
- weeks: 52
- versions: 3
- promotions: 1
- rejections: 1
- rollbacks: 0
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
| 23 | 2012-06-10 | v2 | 0.1397 | 0.50 | SCORED |
| 24 | 2012-06-17 | v2 | 0.1325 | 0.50 | SCORED |
| 25 | 2012-06-24 | v2 | 0.1360 | 0.50 | SCORED |
| 26 | 2012-07-01 | v2 | 0.1407 | 0.75 | SCORED |
| 27 | 2012-07-08 | v2 | 0.1628 | 0.75 | SCORED |
| 28 | 2012-07-15 | v2 | 0.1727 | 0.75 | SCORED |
| 29 | 2012-07-22 | v2 | 0.1851 | 0.75 | SCORED |
| 30 | 2012-07-29 | v2 | 0.1773 | 0.50 | SCORED |
| 31 | 2012-08-05 | v2 | 0.1675 | 0.75 | SCORED |
| 32 | 2012-08-12 | v2 | 0.1679 | 0.75 | SCORED |
| 33 | 2012-08-19 | v2 | 0.1559 | 0.75 | SCORED |
| 34 | 2012-08-26 | v2 | 0.1578 | 0.75 | SCORED |
| 35 | 2012-09-02 | v2 | 0.1542 | 0.75 | SCORED |
| 36 | 2012-09-09 | v2 | 0.1565 | 0.75 | SCORED |
| 37 | 2012-09-16 | v2 | 0.1494 | 0.75 | SCORED |
| 38 | 2012-09-23 | v2 | 0.1457 | 0.75 | SCORED |
| 39 | 2012-09-30 | v2 | 0.1446 | 0.50 | SCORED |
| 40 | 2012-10-07 | v2 | 0.1455 | 0.50 | SCORED |
| 41 | 2012-10-14 | v2 | 0.1635 | 0.50 | SCORED |
| 42 | 2012-10-21 | v2 | 0.1732 | 0.50 | SCORED |
| 43 | 2012-10-28 | v2 | 0.1777 | 0.50 | SCORED |
| 44 | 2012-11-04 | v2 | 0.1906 | 0.75 | SCORED |
| 45 | 2012-11-11 | v2 | 0.1993 | 0.75 | SCORED |
| 46 | 2012-11-18 | v2 | 0.2197 | 0.75 | SCORED |
| 47 | 2012-11-25 | v2 | 0.2690 | 0.75 | SCORED |
| 48 | 2012-12-02 | v2 | 0.2697 | 0.75 | SCORED |
| 49 | 2012-12-09 | v2 | 0.2777 | 0.50 | SCORED |
| 50 | 2012-12-16 | v2 | 0.2808 | 0.50 | SCORED |
| 51 | 2012-12-23 | v2 | 0.2612 | 0.50 | SCORED |
| 52 | 2012-12-30 | v2 | 0.3152 | 0.50 | REJECTED |

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

### tick 52 - v3 - REJECTED

| gate | passed | observed | threshold |
| --- | --- | --- | --- |
| DATA_VALID | True | WARN | not FAIL |
| ABS_WAPE | False | 0.4536 | 0.3000 |
| BEATS_BASELINE | True | 78.1697 | 113.6874 |
| IMPROVEMENT | False | -0.0886 | 0.0500 |
| SIGNIFICANCE | False | -2994.6286 | 0.0000 |
| SEGMENT | True | 0.0959 | 0.1000 |

## Summary

| metric | as-served | frozen v1 |
| --- | --- | --- |
| RMSE | 70.7197 | 109.9249 |
| MAE | 45.8888 | 74.5220 |
| WAPE | 0.1949 | 0.3165 |
| rows | 8686 | 8686 |

as-served RMSE is 0.64x the frozen-v1 RMSE.

promotions=1 rejections=1 rollbacks=0 quarantined=0
