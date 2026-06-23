# Transient Compass

`Transient Compass` is a JUCE-based transient shaper for real DAW workflows.
It analyzes a live mono mix using the framework we built in
[`SpacekidLabs/representation-fragility-lab`](https://github.com/SpacekidLabs/representation-fragility-lab)
and then shapes the audio in place so you can hear the result immediately.

The effect is built around the same framework we developed in the lab:

- the original 10-descriptor representation space
- the same PCA region map
- the same safety assumptions for `stft`, `acf`, `cepstrum`, `cqt`, and `wavelet`
- an adaptive transient score that drives the shaping envelope

## What You Get

- VST3, AU, and Standalone plugin targets
- a live visual map of the current `(z1, z2)` state
- region labels such as `transition_zone`, `periodic_harmonic`, and `transient_overloaded`
- adjustable sensitivity, reset floor, shape time, attack boost, sustain cut, mix, and output gain
- automatic or manual analysis window selection
- a sound-first workflow for auditioning the detector directly on drums

## Build

This repo expects a local JUCE checkout.

Set `JUCE_DIR` if your JUCE folder is not at the default path:

```bash
export JUCE_DIR=/path/to/JUCE
```

Then configure and build with CMake:

```bash
/Applications/CMake.app/Contents/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
/Applications/CMake.app/Contents/bin/cmake --build build --config Release --target TransientCompass
```

On macOS the build produces these bundles under `build/TransientCompass_artefacts/Release/`:

- `VST3/TransientCompass.vst3`
- `AU/TransientCompass.component`
- `Standalone/TransientCompass.app`

## UI

The plugin UI shows:

- the live PCA point on the original framework map
- score and confidence meters
- the current semantic region
- the active primary representation
- the recommended window size
- the current shaper control values

The visual panel is meant to make the detector explainable while still being
usable in a DAW session.
