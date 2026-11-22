from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _vector3(value: tuple[float, float, float], name: str, *, nonnegative: bool) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    if nonnegative and np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return vector


@dataclass(frozen=True)
class PoseEstimatorConfig:
    """Timing and error model for a sampled relative-position estimator."""

    update_period_s: float = 0.02
    latency_s: float = 0.0
    dropout_probability: float = 0.0
    noise_std_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    correlation_time_s: float = 0.0

    def __post_init__(self) -> None:
        scalars = (self.update_period_s, self.latency_s, self.correlation_time_s)
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("pose-estimator timing values must be finite")
        if self.update_period_s <= 0.0:
            raise ValueError("update_period_s must be positive")
        if self.latency_s < 0.0 or self.correlation_time_s < 0.0:
            raise ValueError("latency_s and correlation_time_s must be nonnegative")
        if not np.isfinite(self.dropout_probability) or not (
            0.0 <= self.dropout_probability <= 1.0
        ):
            raise ValueError("dropout_probability must be in [0, 1]")
        _vector3(self.noise_std_m, "noise_std_m", nonnegative=True)
        _vector3(self.bias_m, "bias_m", nonnegative=False)


@dataclass(frozen=True)
class PoseEstimate:
    relative_position_m: FloatArray
    measurement_time_s: float
    age_s: float
    delivered_update: bool
    samples: int
    dropouts: int
    delivered_updates: int


class DelayedPoseEstimator:
    """Sample, delay, and hold relative-position measurements.

    Measurement error follows a stationary first-order Gauss-Markov process. Dropped
    frames never enter the delay queue, so the last delivered estimate is held.
    Reset provides one warm-start estimate; subsequent observations obey configured
    update, latency, dropout, bias, and correlation settings.
    """

    def __init__(self, config: PoseEstimatorConfig) -> None:
        self.config = config
        self._pending: deque[tuple[float, float, FloatArray]] = deque()
        self._estimate_m = np.zeros(3, dtype=float)
        self._measurement_time_s = 0.0
        self._last_sample_time_s = 0.0
        self._next_sample_time_s = 0.0
        self._error_m = np.zeros(3, dtype=float)
        self._samples = 0
        self._dropouts = 0
        self._delivered_updates = 0
        self._initialized = False

    @staticmethod
    def _position(value: FloatArray) -> FloatArray:
        position = np.asarray(value, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"relative position must have shape (3,), got {position.shape}")
        if not np.all(np.isfinite(position)):
            raise ValueError("relative position must contain finite values")
        return position

    def reset(
        self,
        relative_position_m: FloatArray,
        time_s: float,
        rng: np.random.Generator,
    ) -> PoseEstimate:
        position = self._position(relative_position_m)
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be nonnegative and finite")
        noise_std = _vector3(self.config.noise_std_m, "noise_std_m", nonnegative=True)
        self._error_m = rng.normal(0.0, noise_std)
        self._estimate_m = position + np.asarray(self.config.bias_m) + self._error_m
        self._measurement_time_s = float(time_s)
        self._last_sample_time_s = float(time_s)
        self._next_sample_time_s = float(time_s) + self.config.update_period_s
        self._pending.clear()
        self._samples = 1
        self._dropouts = 0
        self._delivered_updates = 1
        self._initialized = True
        return self._snapshot(time_s, delivered_update=True)

    def _advance_error(self, sample_time_s: float, rng: np.random.Generator) -> None:
        noise_std = np.asarray(self.config.noise_std_m, dtype=float)
        elapsed = max(0.0, sample_time_s - self._last_sample_time_s)
        if self.config.correlation_time_s > 0.0:
            correlation = float(np.exp(-elapsed / self.config.correlation_time_s))
            innovation_scale = noise_std * np.sqrt(max(0.0, 1.0 - correlation**2))
            self._error_m = correlation * self._error_m + rng.normal(0.0, innovation_scale)
        else:
            self._error_m = rng.normal(0.0, noise_std)

    def _snapshot(self, time_s: float, *, delivered_update: bool) -> PoseEstimate:
        return PoseEstimate(
            relative_position_m=self._estimate_m.copy(),
            measurement_time_s=self._measurement_time_s,
            age_s=max(0.0, float(time_s) - self._measurement_time_s),
            delivered_update=delivered_update,
            samples=self._samples,
            dropouts=self._dropouts,
            delivered_updates=self._delivered_updates,
        )

    def observe(
        self,
        relative_position_m: FloatArray,
        time_s: float,
        rng: np.random.Generator,
    ) -> PoseEstimate:
        if not self._initialized:
            return self.reset(relative_position_m, time_s, rng)
        position = self._position(relative_position_m)
        if not np.isfinite(time_s) or time_s < self._last_sample_time_s:
            raise ValueError("pose-estimator time must be finite and monotonic")

        tolerance = 1e-12
        if time_s + tolerance >= self._next_sample_time_s:
            self._advance_error(float(time_s), rng)
            self._samples += 1
            if rng.random() < self.config.dropout_probability:
                self._dropouts += 1
            else:
                measurement = position + np.asarray(self.config.bias_m) + self._error_m
                self._pending.append(
                    (
                        float(time_s) + self.config.latency_s,
                        float(time_s),
                        measurement.copy(),
                    )
                )
            self._last_sample_time_s = float(time_s)
            self._next_sample_time_s = float(time_s) + self.config.update_period_s

        delivered = False
        while self._pending and self._pending[0][0] <= time_s + tolerance:
            _, measurement_time_s, measurement = self._pending.popleft()
            self._estimate_m = measurement
            self._measurement_time_s = measurement_time_s
            self._delivered_updates += 1
            delivered = True
        return self._snapshot(time_s, delivered_update=delivered)
