# Adaptive Transient Detector

This repo is a small standalone **tool** for finding transients in audio.
It takes the lessons from the earlier project and turns them into one
practical command-line utility.

## What it does

The tool watches audio frame by frame and blends three signals:

- `spectral flux` for fast attack motion
- `ACF periodicity drop` for loss of stable harmonic structure
- `cepstral disruption` for harmonic-shape change and noise-floor rise

Instead of trusting one representation all the time, it:

- measures confidence per branch
- adapts fusion weights from the current signal state
- switches analysis window size based on whether the frame looks transient,
  stable, or ambiguous

## Quick start

```bash
pip install -e .
```

Run it on a WAV file:

```bash
atd path/to/audio.wav
```

Or let it generate a built-in demo signal:

```bash
atd
```

The output lists the strongest transient frames with their score, confidence,
and the window size the tool would choose next.

## Python use

```python
import numpy as np
from adaptive_transient_detector import AdaptiveTransientDetector

sr = 16000
t = np.arange(sr) / sr
audio = 0.2 * np.sin(2 * np.pi * 220 * t)
audio[7000:7010] += 1.0  # simple transient

detector = AdaptiveTransientDetector(sample_rate=sr)
results = detector.detect(audio)

best = max(results, key=lambda r: r.transient_score)
print(best.sample_index, best.transient_score, best.recommended_window)
```

## Design goals

- stay lightweight and easy to inspect
- work in streaming or offline mode
- keep the tool explainable, not just accurate
- keep the first implementation small enough to iterate quickly
