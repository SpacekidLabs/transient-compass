from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from .config import DetectorConfig
from .detector import AdaptiveTransientDetector


def _build_demo_signal(sample_rate: int) -> np.ndarray:
    t = np.arange(sample_rate) / sample_rate
    audio = 0.18 * np.sin(2 * np.pi * 220.0 * t)
    fade = np.ones_like(audio)
    attack = int(0.05 * sample_rate)
    release = int(0.05 * sample_rate)
    fade[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    fade[-release:] = np.linspace(1.0, 0.0, release, endpoint=False)
    audio *= fade
    audio[int(0.42 * sample_rate) : int(0.42 * sample_rate) + 20] += 1.0
    audio[int(0.73 * sample_rate) : int(0.73 * sample_rate) + 15] -= 0.8
    return audio.astype(np.float64)


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return int(sample_rate), data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive transient detector tool")
    parser.add_argument("audio", nargs="?", help="Path to a WAV file. If omitted, a demo signal is used.")
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--include-warmup", action="store_true", help="Also report hits from the initial warm-up window.")
    parser.add_argument("--json", action="store_true", help="Print decisions as JSON.")
    args = parser.parse_args(argv)

    if args.audio:
        sample_rate, audio = _read_wav(Path(args.audio))
    else:
        sample_rate = 16000
        audio = _build_demo_signal(sample_rate)

    detector = AdaptiveTransientDetector(
        DetectorConfig(
            sample_rate=sample_rate,
            analysis_window=args.frame_length,
            hop_length=args.hop_length,
            transient_threshold=args.threshold,
        )
    )

    results = detector.detect(audio, frame_length=args.frame_length, hop_length=args.hop_length)
    hits = [
        r
        for r in results
        if r.is_transient and (args.include_warmup or r.sample_index >= args.frame_length)
    ]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "sample_index": r.sample_index,
                        "transient_score": r.transient_score,
                        "confidence": r.confidence,
                        "recommended_window": r.recommended_window,
                        "active_branch": r.active_branch,
                    }
                    for r in hits
                ],
                indent=2,
            )
        )
    else:
        if not hits:
            print("No transients found.")
        for r in hits:
            print(
                f"{r.sample_index:8d}  score={r.transient_score:.3f}  "
                f"conf={r.confidence:.3f}  window={r.recommended_window:4d}  branch={r.active_branch}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
