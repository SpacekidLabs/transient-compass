import numpy as np

from adaptive_transient_detector import AdaptiveTransientDetector, DetectorConfig


def make_signal(sr: int = 16000) -> np.ndarray:
    t = np.arange(sr) / sr
    tone = 0.15 * np.sin(2 * np.pi * 220.0 * t)
    tone[7000:7015] += 1.2
    return tone


def test_transient_peak_is_detected():
    detector = AdaptiveTransientDetector(DetectorConfig(sample_rate=16000, analysis_window=1024, hop_length=128))
    results = detector.detect(make_signal())

    assert results, "expected at least one frame decision"
    peak = max(results, key=lambda r: r.transient_score)
    assert peak.transient_score > 0.45
    assert peak.is_transient


def test_recommended_window_shrinks_on_strong_transient():
    detector = AdaptiveTransientDetector(DetectorConfig(sample_rate=16000, analysis_window=1024, hop_length=128))
    results = detector.detect(make_signal())
    peak = max(results, key=lambda r: r.transient_score)
    assert peak.recommended_window <= 512

