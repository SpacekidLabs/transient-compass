from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable

import numpy as np

from .config import DetectorConfig
from .features import BranchFeatures, extract_branch_features


@dataclass(slots=True)
class FrameDecision:
    sample_index: int
    transient_score: float
    confidence: float
    is_transient: bool
    recommended_window: int
    active_branch: str
    branch_features: BranchFeatures
    fusion_weights: dict[str, float]


class AdaptiveTransientDetector:
    """Streaming-friendly transient detector with adaptive representation routing."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._prev_frame: np.ndarray | None = None
        self._sample_cursor = 0
        self._flux_ema = 0.0
        self._rms_ema = 0.0
        self._history: deque[float] = deque(maxlen=self.config.history_limit)

    @property
    def recommended_window(self) -> int:
        if not self._history:
            return self.config.analysis_window
        recent = float(np.mean(self._history))
        if recent >= self.config.strong_transient_threshold:
            return self.config.short_windows[0]
        if recent >= self.config.transient_threshold:
            return self.config.short_windows[min(1, len(self.config.short_windows) - 1)]
        if recent <= 0.30:
            return self.config.long_windows[-1]
        return self.config.analysis_window

    def _adapt_weights(self, feats: BranchFeatures) -> dict[str, float]:
        flux_gate = max(feats.spectral_flux, 0.0) * (0.75 + 0.25 * feats.spectral_confidence)
        acf_gate = feats.acf_instability * (0.65 + 0.35 * feats.acf_confidence)
        cep_gate = feats.cepstral_disruption * (0.60 + 0.40 * feats.cepstral_confidence)
        energy_gate = feats.energy_rise * (0.70 + 0.30 * feats.broadband_peakiness)

        weights = {
            "spectral_flux": flux_gate + self.config.adaptive_blend_bias,
            "acf_drop": acf_gate + self.config.adaptive_blend_bias,
            "cepstral_drop": cep_gate + self.config.adaptive_blend_bias,
            "energy_rise": energy_gate + self.config.adaptive_blend_bias,
        }
        total = sum(weights.values()) + self.config.eps
        return {k: float(v / total) for k, v in weights.items()}

    def _score_frame(self, feats: BranchFeatures) -> tuple[float, float, dict[str, float], str]:
        weights = self._adapt_weights(feats)

        branch_scores = {
            "spectral_flux": float(np.clip(feats.spectral_flux, 0.0, 1.5)),
            "acf_drop": float(np.clip(feats.acf_instability, 0.0, 1.0)),
            "cepstral_drop": float(np.clip(feats.cepstral_disruption, 0.0, 1.0)),
            "energy_rise": float(np.clip(feats.energy_rise, 0.0, 1.5)),
        }

        weighted_sum = sum(weights[name] * branch_scores[name] for name in weights)
        confidence_pool = float(np.mean([
            feats.spectral_confidence,
            feats.acf_confidence,
            feats.cepstral_confidence,
        ]))
        transient_score = float(np.clip(weighted_sum * (0.35 + 0.65 * confidence_pool), 0.0, 1.0))
        active_branch = max(branch_scores, key=branch_scores.get)
        return transient_score, float(confidence_pool), weights, active_branch

    def process_frame(
        self,
        frame: np.ndarray,
        sample_index: int | None = None,
        advance: int | None = None,
    ) -> FrameDecision:
        frame = np.asarray(frame, dtype=np.float64).flatten()
        if frame.size == 0:
            raise ValueError("frame must contain at least one sample")

        rms = float(np.sqrt(np.mean(np.square(frame))))
        self._rms_ema = (1.0 - self.config.rms_ema_alpha) * self._rms_ema + self.config.rms_ema_alpha * rms

        feats = extract_branch_features(
            frame=frame,
            prev_frame=self._prev_frame,
            config=self.config,
            rms_ema=self._rms_ema,
            prev_flux_ema=self._flux_ema,
        )
        score, confidence, weights, active_branch = self._score_frame(feats)
        self._flux_ema = (1.0 - self.config.flux_ema_alpha) * self._flux_ema + self.config.flux_ema_alpha * feats.spectral_flux
        self._history.append(score)
        self._prev_frame = frame.copy()

        if sample_index is None:
            self._sample_cursor += advance if advance is not None else frame.size
            decision_index = self._sample_cursor
        else:
            decision_index = sample_index
            self._sample_cursor = sample_index

        if score >= self.config.strong_transient_threshold:
            recommended_window = self.config.short_windows[0]
        elif score >= self.config.transient_threshold:
            recommended_window = self.config.short_windows[min(1, len(self.config.short_windows) - 1)]
        elif confidence <= self.config.confidence_floor:
            recommended_window = self.config.long_windows[min(1, len(self.config.long_windows) - 1)]
        else:
            recommended_window = self.recommended_window

        return FrameDecision(
            sample_index=decision_index,
            transient_score=score,
            confidence=confidence,
            is_transient=score >= self.config.transient_threshold,
            recommended_window=recommended_window,
            active_branch=active_branch,
            branch_features=feats,
            fusion_weights=weights,
        )

    def detect(self, audio: np.ndarray, frame_length: int | None = None, hop_length: int | None = None) -> list[FrameDecision]:
        audio = np.asarray(audio, dtype=np.float64).flatten()
        if audio.size == 0:
            return []

        frame_length = frame_length or self.config.analysis_window
        hop_length = hop_length or self.config.hop_length
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("frame_length and hop_length must be positive")
        if audio.size < frame_length:
            pad = frame_length - audio.size
            audio = np.pad(audio, (0, pad))

        decisions: list[FrameDecision] = []
        for start in range(0, audio.size - frame_length + 1, hop_length):
            frame = audio[start : start + frame_length]
            center_index = start + (frame_length // 2)
            decisions.append(self.process_frame(frame, sample_index=center_index, advance=hop_length))
        return decisions

    def reset(self) -> None:
        self._prev_frame = None
        self._sample_cursor = 0
        self._flux_ema = 0.0
        self._rms_ema = 0.0
        self._history.clear()
