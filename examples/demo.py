from __future__ import annotations

import numpy as np

from adaptive_transient_detector import AdaptiveTransientDetector


def main() -> None:
    sr = 16000
    t = np.arange(sr) / sr
    audio = 0.18 * np.sin(2 * np.pi * 220.0 * t)
    audio[6800:6820] += 1.0

    detector = AdaptiveTransientDetector()
    results = detector.detect(audio)
    peak = max(results, key=lambda r: r.transient_score)

    print("peak sample:", peak.sample_index)
    print("score:", round(peak.transient_score, 3))
    print("confidence:", round(peak.confidence, 3))
    print("recommended_window:", peak.recommended_window)
    print("active_branch:", peak.active_branch)


if __name__ == "__main__":
    main()

