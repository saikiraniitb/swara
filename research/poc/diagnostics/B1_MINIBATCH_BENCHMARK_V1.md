# B1 Mini-Batch Runtime Benchmark

Benchmark-only. No B1 training was performed.

## Batch size 2

- mean/median/p90 seconds/step: `0.0813` / `0.0788` / `0.1112`
- mean padded / real frames per batch: `694.7` / `683.6`
- padding percentage: `1.6%`
- real frames/sec: `8405.3`
- utterances/sec: `24.59`
- steps/epoch: `16`
- estimated 10/20/40-epoch wall time (minutes): `0.22` / `0.43` / `0.87`

## Batch size 4

- mean/median/p90 seconds/step: `0.1460` / `0.1293` / `0.1980`
- mean padded / real frames per batch: `1488.8` / `1401.3`
- padding percentage: `5.9%`
- real frames/sec: `9596.9`
- utterances/sec: `27.39`
- steps/epoch: `8`
- estimated 10/20/40-epoch wall time (minutes): `0.19` / `0.39` / `0.78`

## Batch size 8

- mean/median/p90 seconds/step: `0.3175` / `0.2776` / `0.4983`
- mean padded / real frames per batch: `3342.0` / `2926.0`
- padding percentage: `12.4%`
- real frames/sec: `9215.8`
- utterances/sec: `25.20`
- steps/epoch: `4`
- estimated 10/20/40-epoch wall time (minutes): `0.21` / `0.42` / `0.85`

## MPS probe (batch size 4)

Status: `PASS`

- seconds/step: `0.3679`
- real frames/sec: `4141.9`
- fallback warnings: ['none observed']
