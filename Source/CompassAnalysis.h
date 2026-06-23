#pragma once

#include <JuceHeader.h>
#include <array>
#include <vector>

enum class CompassRegion
{
    TransitionZone = 0,
    PeriodicHarmonic,
    NoiseCollapse,
    TransientOverloaded,
    SmoothLowpass
};

struct CompassAnalysisResult
{
    float score = 0.0f;
    float confidence = 0.0f;
    float z1 = 0.0f;
    float z2 = 0.0f;
    float spectralFlux = 0.0f;
    float acfDrop = 0.0f;
    float cepstralDrop = 0.0f;
    float energyRise = 0.0f;
    bool isTransient = false;
    int recommendedWindow = 2048;
    juce::String region = "transition_zone";
    juce::String primaryRepresentation = "stft";
};

class CompassAnalysisEngine
{
public:
    CompassAnalysisEngine() = default;

    void prepare (double sampleRate, int maxFrameSize = 4096);
    void reset();

    CompassAnalysisResult analyseFrame (const float* samples,
                                        int numSamples,
                                        float transientThreshold,
                                        float resetThreshold,
                                        int warmupFrames,
                                        int refractoryFrames,
                                        bool adaptiveWindow);

    int getCurrentWindowSize() const { return currentWindowSize; }

private:
    static int nextPowerOfTwo (int value);
    static float computeHannWindow (int n, int i);
    static float clamp01 (float v);
    static juce::String regionToString (CompassRegion region);

    void ensureFFTSize (int size);
    std::vector<float> computeMagnitudeSpectrum (const float* samples, int numSamples);
    std::vector<float> computeACF (const float* samples, int numSamples) const;
    std::vector<float> computeCepstrum (const std::vector<float>& magnitude) const;
    float computeRMS (const float* samples, int numSamples) const;
    float computeSpectralEntropy (const std::vector<float>& mag) const;
    float computeSpectralFlatness (const std::vector<float>& mag) const;
    float computeZCR (const float* frame, int numSamples) const;
    std::pair<float, float> computeACFFeatures (const std::vector<float>& acf, double effectiveSampleRate) const;
    float computeCrestFactor (const float* frame, int numSamples, float rms) const;
    float computeSpectralRolloff (const std::vector<float>& mag) const;
    float computeHoyerSparsity (const std::vector<float>& mag) const;
    float computeKurtosis (const float* frame, int numSamples, float rms) const;
    float computeModulation (const float* frame, int numSamples) const;

    CompassRegion classifyRegion (float z1, float z2) const;

    int chooseRecommendedWindow (CompassRegion region, bool adaptiveWindow) const;

    double sampleRate = 48000.0;
    int maxFrameSize = 4096;
    int currentFFTSize = 1024;
    int currentWindowSize = 1024;

    int framesSeen = 0;
    int refractoryCounter = 0;
    bool eventActive = false;

    std::vector<float> previousFrame;
    std::vector<float> previousMagnitude;
    std::vector<float> previousACF;
    std::vector<float> previousCepstrum;

    float previousFeatureFlux = 0.0f;
    float previousHarmonicRatio = 0.0f;
    float previousACFPeriodicity = 0.0f;
    float previousCepstrumPeak = 0.0f;
    float previousRms = 0.0f;

    std::unique_ptr<juce::dsp::FFT> fft;
    std::vector<float> fftBuffer;
    std::vector<float> cepstrumBuffer;
};
