from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DetectorConfig:
    """Configuration for the adaptive transient detector."""

    sample_rate: int = 16000
    analysis_window: int = 1024
    hop_length: int = 256
    min_period_hz: float = 80.0
    max_period_hz: float = 1200.0
    transient_threshold: float = 0.58
    strong_transient_threshold: float = 0.74
    confidence_floor: float = 0.18
    flux_ema_alpha: float = 0.25
    rms_ema_alpha: float = 0.20
    short_windows: tuple[int, ...] = (256, 512, 1024)
    long_windows: tuple[int, ...] = (1024, 2048, 4096)
    cepstrum_floor_db: float = -40.0
    cepstrum_peak_db: float = -10.0
    adaptive_blend_bias: float = 0.08
    history_limit: int = 12
    eps: float = 1e-9
    metadata: dict[str, float] = field(default_factory=dict)

