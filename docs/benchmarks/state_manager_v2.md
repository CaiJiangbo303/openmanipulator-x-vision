# Object State Manager V2.0

## Architecture

- Default backend: Alpha-Beta.
- Optional backend: Adaptive CV-KF.
- Adaptive backend requires TimestampGuard.
- Independent filter state per generation.
- Fast3D drives the motion filter.
- Mask3D bias correction remains outside the KF state.
- Existing conservative uncertainty policy retained.
- ZUPT is not enabled.

## Adaptive CV-KF parameters (SI)

- q_low: 1e-4 m^2/s^3
- q_high: 1e-3 m^2/s^3
- measurement_std: 2e-4 m
- NIS threshold: 11.345
- boost duration: 0.20 s
- Joseph covariance update

## Validation

- Adaptive filter unit tests: 10/10 PASS
- TimestampGuard unit tests: 14/14 PASS
- Backend integration tests: 8/8 PASS
- Adaptive Gate ROS E2E: 15/15 PASS
- Numerical equivalence: 4986 updates PASS
- High-Q decision mismatches: 0

## Real static A/B benchmark

Duration: 60 seconds.

| Metric | Alpha-Beta | Adaptive CV-KF |
|---|---:|---:|
| Valid rate (%) | 98.192 | 98.192 |
| Output rate (Hz) | 30.422 | 30.422 |
| Mean speed (mm/s) | 2.1523 | 1.3816 |
| Speed P95 (mm/s) | 3.8399 | 2.4803 |
| Position std X (mm) | 0.2276 | 0.1955 |
| Position std Y (mm) | 0.4887 | 0.4758 |
| Position std Z (mm) | 0.6824 | 0.6824 |
| Measurement age P95 (ms) | 74.655 | 75.057 |

## Real motion prediction benchmark

Duration: 85 seconds.
Common state source keys: 2099.

Mask3D reference, motion phases only.
These are observation-reference residuals,
not independent ground-truth errors.

| Horizon | Alpha-Beta RMSE (mm) | Adaptive RMSE (mm) |
|---|---:|---:|
| 100 ms | 3.205 | 3.238 |
| 200 ms | 6.371 | 6.617 |
| 300 ms | 9.379 | 9.694 |

## Decision

Retain Alpha-Beta as the default backend.

Adaptive CV-KF reduces static velocity noise,
but has not demonstrated superior real dynamic
prediction performance.

Keep Adaptive CV-KF as an experimental backend.
Freeze the current filter parameters.

## Remaining limitations

- WSL and camera timestamp discontinuities persist.
- Z-axis output jitter exceeds the 0.5 mm target.
- Mask3D correction jitter requires investigation.
- Measurement covariance is not fully calibrated.
- Long-horizon motion prediction is unreliable.
- No physical robot motion or stopping validation.
- State Gate authorization is perception-only.
