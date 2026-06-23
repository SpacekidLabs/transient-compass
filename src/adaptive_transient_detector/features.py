from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig


@dataclass(slots=True)
class BranchFeatures:
    spectral_flux: float
    spectral_confidence: float
    acf_instability: float
    acf_confidence: float
    cepstral_disruption: float
    cepstral_confidence: float
    energy_rise: float
    broadband_peakiness: float


def _safe_rfft_mag(frame: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(frame))


def _hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * idx) / (n - 1))


def compute_normalized_acf(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float64).flatten()
    if x.size == 0:
        return np.zeros(1, dtype=np.float64)

    x = x - np.mean(x)
    energy = np.dot(x, x)
    if energy <= 0.0:
        return np.zeros(1, dtype=np.float64)

    corr = np.correlate(x, x, mode="full")[x.size - 1 :]
    return corr / (energy + 1e-12)


def compute_real_cepstrum(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float64).flatten()
    if x.size == 0:
        return np.zeros(1, dtype=np.float64)

    spectrum = np.fft.rfft(x * _hann_window(x.size))
    log_mag = np.log(np.abs(spectrum) + 1e-12)
    return np.fft.irfft(log_mag)


def _range_slice(sr: int, min_hz: float, max_hz: float, n: int) -> slice:
    lo = max(1, int(sr / max_hz))
    hi = min(n, int(sr / min_hz))
    return slice(lo, max(lo + 1, hi))


def extract_branch_features(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    config: DetectorConfig,
    rms_ema: float,
    prev_flux_ema: float,
) -> BranchFeatures:
    x = np.asarray(frame, dtype=np.float64).flatten()
    if x.size == 0:
        raise ValueError("frame must contain at least one sample")

    x = x * _hann_window(x.size)
    mag = _safe_rfft_mag(x)

    if prev_frame is None:
        prev_mag = np.zeros_like(mag)
    else:
        prev = np.asarray(prev_frame, dtype=np.float64).flatten()
        prev = prev[: x.size] if prev.size >= x.size else np.pad(prev, (0, x.size - prev.size))
        prev = prev * _hann_window(x.size)
        prev_mag = _safe_rfft_mag(prev)

    flux = float(np.sum(np.maximum(mag - prev_mag, 0.0)))
    flux_norm = flux / (np.sum(prev_mag) + np.sum(mag) + config.eps)
    if prev_frame is None:
        spectral_flux = 0.0
    else:
        flux_ema = (1.0 - config.flux_ema_alpha) * prev_flux_ema + config.flux_ema_alpha * flux_norm
        spectral_flux = float(np.clip((flux_norm - flux_ema) / (flux_ema + config.eps), 0.0, 1.5))
    spectral_confidence = float(np.clip((np.max(mag) / (np.sum(mag) + config.eps) - 0.02) / 0.25, 0.0, 1.0))

    acf = compute_normalized_acf(x)
    lag_slice = _range_slice(config.sample_rate, config.min_period_hz, config.max_period_hz, acf.size)
    acf_region = acf[lag_slice]
    if acf_region.size == 0:
        peak_ratio = 0.0
    else:
        peak_ratio = float(np.max(acf_region))
    acf_confidence = float(np.clip((peak_ratio - 0.12) / 0.68, 0.0, 1.0))

    if prev_frame is None:
        acf_instability = 0.0
    else:
        prev_acf = compute_normalized_acf(prev_frame[: x.size] if prev_frame.size >= x.size else np.pad(prev_frame, (0, x.size - prev_frame.size)))
        prev_acf_region = prev_acf[lag_slice] if prev_acf.size else np.zeros(1, dtype=np.float64)
        prev_peak_ratio = float(np.max(prev_acf_region)) if prev_acf_region.size else 0.0
        acf_instability = float(np.clip((prev_peak_ratio - peak_ratio) / 0.30, 0.0, 1.0))

    cep = compute_real_cepstrum(x)
    q_slice = _range_slice(config.sample_rate, config.min_period_hz, config.max_period_hz, cep.size)
    cep_region = np.abs(cep[q_slice])
    cep_peak = float(np.max(cep_region)) if cep_region.size else 0.0
    cep_floor = float(np.abs(cep[0]))
    cepstral_confidence = float(np.clip((cep_floor - config.cepstrum_floor_db) / (config.cepstrum_peak_db - config.cepstrum_floor_db), 0.0, 1.0))

    if prev_frame is None:
        cepstral_disruption = 0.0
    else:
        prev_cep = compute_real_cepstrum(prev_frame[: x.size] if prev_frame.size >= x.size else np.pad(prev_frame, (0, x.size - prev_frame.size)))
        prev_cep_region = np.abs(prev_cep[q_slice]) if prev_cep.size else np.zeros(1, dtype=np.float64)
        prev_cep_peak = float(np.max(prev_cep_region)) if prev_cep_region.size else 0.0
        prev_cep_floor = float(np.abs(prev_cep[0])) if prev_cep.size else 0.0
        peak_drop = 0.0 if prev_cep_peak <= config.eps else max(0.0, (prev_cep_peak - cep_peak) / (prev_cep_peak + config.eps))
        floor_rise = 0.0 if prev_cep_floor <= config.eps else max(0.0, (cep_floor - prev_cep_floor) / (prev_cep_floor + config.eps))
        cepstral_disruption = float(np.clip(0.7 * peak_drop + 0.3 * floor_rise, 0.0, 1.0))

    rms = float(np.sqrt(np.mean(np.square(x))))
    if prev_frame is None or rms_ema <= config.eps:
        energy_rise = 0.0
    else:
        energy_rise = float(np.clip((rms - rms_ema) / (rms_ema + config.eps) + 0.25, 0.0, 1.5))
    broadband_peakiness = float(np.clip((np.max(mag) / (np.mean(mag) + config.eps) - 1.0) / 12.0, 0.0, 1.0))

    return BranchFeatures(
        spectral_flux=spectral_flux,
        spectral_confidence=spectral_confidence,
        acf_instability=acf_instability,
        acf_confidence=acf_confidence,
        cepstral_disruption=cepstral_disruption,
        cepstral_confidence=cepstral_confidence,
        energy_rise=energy_rise,
        broadband_peakiness=broadband_peakiness,
    )
