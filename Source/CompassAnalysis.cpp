#include "CompassAnalysis.h"

#include <algorithm>
#include <cmath>

namespace
{
static constexpr std::array<float, 10> mu
{
    0.76195118f, 0.29042404f, 0.11610899f, 0.63087928f, 2.64905438f,
    0.56809152f, 3702.86576f, 0.71430379f, 2.86565869f, 0.05771320f
};

static constexpr std::array<float, 10> sigma
{
    0.08931102f, 0.19573561f, 0.16632286f, 0.25154298f, 0.96131249f,
    0.10439500f, 2717.09600f, 0.16113735f, 1.62378944f, 0.05117988f
};

static constexpr std::array<std::array<float, 10>, 2> vt
{
    std::array<float, 10>{ 0.47315060f, 0.46611747f, 0.32685922f, -0.12470205f, 0.11155174f, -0.01194378f, 0.44399277f, -0.46931129f, 0.04403310f, -0.06758288f },
    std::array<float, 10>{ -0.11994855f, -0.16311870f, 0.19125473f, -0.39156500f, 0.52897854f, -0.25676939f, -0.16203983f, -0.06562160f, 0.52857375f, 0.33675064f }
};

static constexpr std::array<float, 6> wStft
{
    0.84399633f, -0.02332773f, -0.03847526f, -0.00716794f, -0.00057247f, 0.00734064f
};

static constexpr std::array<float, 6> wAcf
{
    0.86417094f, -0.00584328f, -0.04004930f, -0.00749728f, -0.00053847f, 0.00706432f
};

static constexpr std::array<float, 6> wCep
{
    0.66975406f, -0.01080766f, -0.02257589f, -0.00841665f, -0.00032442f, 0.00465065f
};

static constexpr std::array<float, 6> wCqt
{
    0.85911548f, -0.01305596f, -0.03705924f, -0.00408812f, -0.00047205f, 0.00600636f
};

static constexpr std::array<float, 6> wCwt
{
    0.88822169f, 0.00137673f, -0.03561487f, -0.00836219f, -0.00049869f, 0.00658515f
};

template <typename ArrayType>
float dot6 (const ArrayType& weights, const std::array<float, 6>& poly)
{
    float sum = 0.0f;
    for (size_t i = 0; i < poly.size(); ++i)
        sum += weights[(size_t) i] * poly[i];
    return sum;
}
} // namespace

int CompassAnalysisEngine::nextPowerOfTwo (int value)
{
    int p = 1;
    while (p < value)
        p <<= 1;
    return jlimit (64, 4096, p);
}

float CompassAnalysisEngine::computeHannWindow (int n, int i)
{
    if (n <= 1)
        return 1.0f;

    return 0.5f * (1.0f - std::cos (juce::MathConstants<float>::twoPi * (float) i / (float) (n - 1)));
}

float CompassAnalysisEngine::clamp01 (float v)
{
    return jlimit (0.0f, 1.0f, v);
}

juce::String CompassAnalysisEngine::regionToString (CompassRegion region)
{
    switch (region)
    {
        case CompassRegion::PeriodicHarmonic:    return "periodic_harmonic";
        case CompassRegion::NoiseCollapse:       return "noise_collapse";
        case CompassRegion::TransientOverloaded: return "transient_overloaded";
        case CompassRegion::SmoothLowpass:       return "smooth_lowpass";
        case CompassRegion::TransitionZone:
        default:                                 return "transition_zone";
    }
}

void CompassAnalysisEngine::prepare (double newSampleRate, int newMaxFrameSize)
{
    sampleRate = newSampleRate;
    maxFrameSize = newMaxFrameSize;
    reset();
}

void CompassAnalysisEngine::reset()
{
    framesSeen = 0;
    refractoryCounter = 0;
    eventActive = false;
    previousFrame.clear();
    previousMagnitude.clear();
    previousACF.clear();
    previousCepstrum.clear();
    previousFeatureFlux = 0.0f;
    previousHarmonicRatio = 0.0f;
    previousACFPeriodicity = 0.0f;
    previousCepstrumPeak = 0.0f;
    previousRms = 0.0f;
    currentWindowSize = 2048;
    ensureFFTSize (currentWindowSize);
}

void CompassAnalysisEngine::ensureFFTSize (int size)
{
    const int fftSize = nextPowerOfTwo (size);
    if (fft && fftSize == currentFFTSize)
        return;

    currentFFTSize = fftSize;
    fft = std::make_unique<juce::dsp::FFT> ((int) std::log2 ((double) currentFFTSize));
    fftBuffer.assign ((size_t) currentFFTSize * 2, 0.0f);
    cepstrumBuffer.assign ((size_t) currentFFTSize * 2, 0.0f);
    previousMagnitude.assign ((size_t) currentFFTSize / 2 + 1, 0.0f);
    previousACF.assign ((size_t) currentFFTSize, 0.0f);
    previousCepstrum.assign ((size_t) currentFFTSize, 0.0f);
}

std::vector<float> CompassAnalysisEngine::computeMagnitudeSpectrum (const float* samples, int numSamples)
{
    ensureFFTSize (numSamples);

    std::fill (fftBuffer.begin(), fftBuffer.end(), 0.0f);
    for (int i = 0; i < numSamples; ++i)
        fftBuffer[(size_t) i] = samples[i] * computeHannWindow (numSamples, i);

    fft->performRealOnlyForwardTransform (fftBuffer.data());

    const int bins = currentFFTSize / 2 + 1;
    std::vector<float> mag ((size_t) bins, 0.0f);
    mag[0] = std::abs (fftBuffer[0]);

    for (int k = 1; k < bins - 1; ++k)
    {
        const float re = fftBuffer[(size_t) 2 * k];
        const float im = fftBuffer[(size_t) 2 * k + 1];
        mag[(size_t) k] = std::sqrt (re * re + im * im);
    }

    mag[(size_t) bins - 1] = std::abs (fftBuffer[(size_t) 1]);
    return mag;
}

std::vector<float> CompassAnalysisEngine::computeACF (const float* samples, int numSamples) const
{
    std::vector<float> acf ((size_t) numSamples, 0.0f);
    for (int lag = 0; lag < numSamples; ++lag)
    {
        float sum = 0.0f;
        for (int i = 0; i < numSamples - lag; ++i)
            sum += samples[i] * samples[i + lag];
        acf[(size_t) lag] = sum;
    }

    const float r0 = acf[0] + 1e-12f;
    for (auto& value : acf)
        value /= r0;

    return acf;
}

std::vector<float> CompassAnalysisEngine::computeCepstrum (const std::vector<float>& magnitude) const
{
    const int bins = (int) magnitude.size();
    if (bins <= 1)
        return {};

    const int fftSize = (bins - 1) * 2;
    std::vector<float> temp ((size_t) fftSize * 2, 0.0f);

    temp[0] = std::log (magnitude[0] + 1e-12f);
    for (int k = 1; k < bins - 1; ++k)
    {
        const float logMag = std::log (magnitude[(size_t) k] + 1e-12f);
        temp[(size_t) 2 * k] = logMag;
        temp[(size_t) 2 * k + 1] = 0.0f;
    }
    temp[(size_t) 1] = std::log (magnitude.back() + 1e-12f);

    juce::dsp::FFT localFft ((int) std::round (std::log2 ((double) fftSize)));
    localFft.performRealOnlyInverseTransform (temp.data());

    std::vector<float> cep ((size_t) bins, 0.0f);
    for (int i = 0; i < bins; ++i)
        cep[(size_t) i] = temp[(size_t) i];
    return cep;
}

float CompassAnalysisEngine::computeRMS (const float* samples, int numSamples) const
{
    double sum = 0.0;
    for (int i = 0; i < numSamples; ++i)
        sum += (double) samples[i] * (double) samples[i];
    return (float) std::sqrt (sum / std::max (1, numSamples));
}

float CompassAnalysisEngine::computeSpectralEntropy (const std::vector<float>& mag) const
{
    float sum = 0.0f;
    for (float m : mag)
        sum += m;

    if (sum < 1e-12f)
        return 1.0f;

    float entropySum = 0.0f;
    for (float m : mag)
    {
        const float p = m / sum;
        if (p > 1e-12f)
            entropySum += p * std::log2 (p);
    }

    const float maxEntropy = std::log2 ((float) mag.size());
    return -entropySum / maxEntropy;
}

float CompassAnalysisEngine::computeSpectralFlatness (const std::vector<float>& mag) const
{
    float sum = 0.0f;
    float sumLog = 0.0f;
    for (float m : mag)
    {
        sum += m;
        sumLog += std::log (m + 1e-12f);
    }

    const float arithmeticMean = sum / (float) mag.size();
    const float geometricMean = std::exp (sumLog / (float) mag.size());
    return geometricMean / (arithmeticMean + 1e-12f);
}

float CompassAnalysisEngine::computeZCR (const float* frame, int numSamples) const
{
    float sumDiff = 0.0f;
    for (int i = 1; i < numSamples; ++i)
    {
        const float sign1 = (frame[i] >= 0.0f) ? 1.0f : -1.0f;
        const float sign0 = (frame[i - 1] >= 0.0f) ? 1.0f : -1.0f;
        sumDiff += std::abs (sign1 - sign0);
    }

    return (sumDiff / (numSamples - 1)) / 2.0f;
}

std::pair<float, float> CompassAnalysisEngine::computeACFFeatures (const std::vector<float>& acf) const
{
    const int minLag = juce::jmax (1, (int) std::floor (sampleRate / 1000.0));
    int maxLag = juce::jmax (minLag + 1, (int) std::floor (sampleRate / 80.0));
    if (maxLag >= (int) acf.size())
        maxLag = (int) acf.size() - 1;

    if (minLag >= maxLag)
        return { 0.0f, 0.0f };

    float maxVal = -1.0f;
    float minVal = 2.0f;
    float acfSum = 0.0f;

    for (int lag = minLag; lag <= maxLag; ++lag)
    {
        const float value = acf[(size_t) lag];
        acfSum += value;
        maxVal = std::max (maxVal, value);
        minVal = std::min (minVal, value);
    }

    const int numLags = maxLag - minLag + 1;
    const float acfMean = acfSum / (float) numLags;
    const float harmonicRatio = maxVal;
    float periodicity = (maxVal - acfMean) / (maxVal - minVal + 1e-10f);
    periodicity = std::clamp (periodicity, 0.0f, 1.0f);

    return { harmonicRatio, periodicity };
}

float CompassAnalysisEngine::computeCrestFactor (const float* frame, int numSamples, float rms) const
{
    float maxVal = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        maxVal = std::max (maxVal, std::abs (frame[i]));
    return maxVal / (rms + 1e-12f);
}

float CompassAnalysisEngine::computeSpectralRolloff (const std::vector<float>& mag) const
{
    float totalSum = 0.0f;
    for (float m : mag)
        totalSum += m;

    if (totalSum < 1e-12f)
        return 0.0f;

    const float threshold = 0.85f * totalSum;
    float runningSum = 0.0f;
    int rolloffBin = 0;
    for (int i = 0; i < (int) mag.size(); ++i)
    {
        runningSum += mag[(size_t) i];
        if (runningSum >= threshold)
        {
            rolloffBin = i;
            break;
        }
    }

    const float binWidth = (float) (sampleRate / (2.0 * (mag.size() - 1)));
    return rolloffBin * binWidth;
}

float CompassAnalysisEngine::computeHoyerSparsity (const std::vector<float>& mag) const
{
    float l1 = 0.0f;
    float l2Sq = 0.0f;
    for (float m : mag)
    {
        l1 += std::abs (m);
        l2Sq += m * m;
    }

    const float l2 = std::sqrt (l2Sq);
    if (l2 < 1e-12f)
        return 1.0f;

    const float n = (float) mag.size();
    const float ratio = l1 / l2;
    const float sqrtN = std::sqrt (n);
    return (sqrtN - ratio) / (sqrtN - 1.0f);
}

float CompassAnalysisEngine::computeKurtosis (const float* frame, int numSamples, float rms) const
{
    juce::ignoreUnused (rms);

    float sumVal = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        sumVal += frame[i];

    const float mean = sumVal / (float) numSamples;
    float sumFourthDiff = 0.0f;
    float sumSqDiff = 1e-12f;

    for (int i = 0; i < numSamples; ++i)
    {
        const float diff = frame[i] - mean;
        sumSqDiff += diff * diff;
        sumFourthDiff += diff * diff * diff * diff;
    }

    const float var = sumSqDiff / (float) numSamples;
    const float fourthMoment = sumFourthDiff / (float) numSamples;
    return fourthMoment / (var * var + 1e-12f);
}

float CompassAnalysisEngine::computeModulation (const float* frame, int numSamples) const
{
    const int subSize = numSamples / 4;
    if (subSize <= 0)
        return 0.0f;

    float rmsVals[4] {};
    for (int q = 0; q < 4; ++q)
    {
        float sumSq = 0.0f;
        for (int i = 0; i < subSize; ++i)
        {
            const float val = frame[q * subSize + i];
            sumSq += val * val;
        }

        rmsVals[q] = std::sqrt (sumSq / subSize);
    }

    const float meanRms = (rmsVals[0] + rmsVals[1] + rmsVals[2] + rmsVals[3]) / 4.0f;
    float sumSqDiff = 0.0f;
    for (int q = 0; q < 4; ++q)
    {
        const float diff = rmsVals[q] - meanRms;
        sumSqDiff += diff * diff;
    }

    return std::sqrt (sumSqDiff / 4.0f);
}

CompassRegion CompassAnalysisEngine::classifyRegion (float z1, float z2) const
{
    if (z1 > 1.5f)
        return CompassRegion::NoiseCollapse;

    if (z1 < -0.5f && z2 < -0.2f)
        return CompassRegion::PeriodicHarmonic;

    if (z2 > 1.5f)
        return CompassRegion::TransientOverloaded;

    if (z1 >= -0.5f && z1 <= 1.5f && z2 < -0.2f)
        return CompassRegion::SmoothLowpass;

    return CompassRegion::TransitionZone;
}

int CompassAnalysisEngine::chooseRecommendedWindow (CompassRegion region, bool adaptiveWindow) const
{
    if (! adaptiveWindow)
        return currentWindowSize;

    switch (region)
    {
        case CompassRegion::NoiseCollapse:       return 4096;
        case CompassRegion::TransientOverloaded: return 1024;
        case CompassRegion::PeriodicHarmonic:    return 2048;
        case CompassRegion::SmoothLowpass:       return 2048;
        case CompassRegion::TransitionZone:
        default:                                 return 2048;
    }
}

CompassAnalysisResult CompassAnalysisEngine::analyseFrame (const float* samples,
                                                           int numSamples,
                                                           float transientThreshold,
                                                           float resetThreshold,
                                                           int warmupFrames,
                                                           int refractoryFrames,
                                                           bool adaptiveWindow)
{
    CompassAnalysisResult result;
    if (samples == nullptr || numSamples <= 0)
        return result;

    currentWindowSize = numSamples;
    ensureFFTSize (numSamples);

    const float rms = computeRMS (samples, numSamples);
    const float zcr = computeZCR (samples, numSamples);
    const float crestFactor = computeCrestFactor (samples, numSamples, rms);
    const float kurtosis = computeKurtosis (samples, numSamples, rms);
    const auto acf = computeACF (samples, numSamples);
    const auto acfFeatures = computeACFFeatures (acf);
    const float harmonicRatio = acfFeatures.first;
    const float periodicity = acfFeatures.second;
    const auto magnitude = computeMagnitudeSpectrum (samples, numSamples);
    const float entropy = computeSpectralEntropy (magnitude);
    const float flatness = computeSpectralFlatness (magnitude);
    const float rolloff = computeSpectralRolloff (magnitude);
    const float sparsity = computeHoyerSparsity (magnitude);
    const float modulation = computeModulation (samples, numSamples);
    const auto cepstrum = computeCepstrum (magnitude);

    std::array<float, 10> features
    {
        entropy,
        flatness,
        zcr,
        harmonicRatio,
        crestFactor,
        periodicity,
        rolloff,
        sparsity,
        kurtosis,
        modulation
    };

    std::array<float, 10> standardized {};
    for (size_t i = 0; i < features.size(); ++i)
        standardized[i] = (features[i] - mu[i]) / sigma[i];

    float z1 = 0.0f;
    float z2 = 0.0f;
    for (size_t j = 0; j < features.size(); ++j)
    {
        z1 += standardized[j] * vt[0][j];
        z2 += standardized[j] * vt[1][j];
    }

    const CompassRegion region = classifyRegion (z1, z2);

    const std::array<float, 6> poly
    {
        1.0f,
        z1,
        z2,
        z1 * z1,
        z2 * z2,
        z1 * z2
    };

    const float stftSafety = clamp01 (dot6 (wStft, poly));
    const float acfSafety = clamp01 (dot6 (wAcf, poly));
    const float cepSafety = clamp01 (dot6 (wCep, poly));
    const float cqtSafety = clamp01 (dot6 (wCqt, poly));
    const float cwtSafety = clamp01 (dot6 (wCwt, poly));

    const std::array<float, 5> safetyScores { stftSafety, acfSafety, cepSafety, cqtSafety, cwtSafety };
    const int primaryIndex = (int) std::distance (safetyScores.begin(),
                                                  std::max_element (safetyScores.begin(), safetyScores.end()));

    const float confidence = (stftSafety + acfSafety + cepSafety) / 3.0f;
    const float transientBias = clamp01 (0.55f * clamp01 ((z2 + 3.0f) / 6.0f)
                                         + 0.20f * (1.0f - stftSafety)
                                         + 0.15f * (1.0f - cepSafety)
                                         + 0.10f * clamp01 (1.0f - flatness));
    const float spectralFlux = clamp01 (previousMagnitude.empty() ? 0.0f : std::abs (transientBias - previousFeatureFlux) * 1.8f);
    const float acfDrop = clamp01 (std::max (0.0f, previousACFPeriodicity - periodicity));
    const float cepstralPeak = cepstrum.empty() ? 0.0f : *std::max_element (cepstrum.begin(), cepstrum.end());
    const float cepstralDrop = clamp01 (std::max (0.0f, previousCepstrumPeak - cepstralPeak) * 2.5f);
    const float energyRise = clamp01 (previousRms <= 1e-9f ? 0.0f : (rms - previousRms) / (previousRms + 1e-9f) + 0.20f);

    const bool regionFeelsTransient = (region == CompassRegion::TransientOverloaded)
                                   || (z2 > 1.25f && stftSafety < 0.82f)
                                   || (spectralFlux > 0.30f && energyRise > 0.12f);

    const float score = clamp01 (0.42f * transientBias
                               + 0.23f * spectralFlux
                               + 0.15f * acfDrop
                               + 0.12f * cepstralDrop
                               + 0.08f * energyRise);

    if (framesSeen < warmupFrames)
    {
        result.isTransient = false;
        result.score = 0.0f;
    }
    else if (eventActive)
    {
        if (score < resetThreshold && ! regionFeelsTransient)
            eventActive = false;
        result.isTransient = false;
    }
    else
    {
        result.isTransient = (score >= transientThreshold && refractoryCounter == 0 && regionFeelsTransient);
        if (result.isTransient)
        {
            eventActive = true;
            refractoryCounter = std::max (0, refractoryFrames);
        }
    }

    if (refractoryCounter > 0)
        --refractoryCounter;

    result.score = score;
    result.confidence = confidence;
    result.z1 = z1;
    result.z2 = z2;
    result.spectralFlux = spectralFlux;
    result.acfDrop = acfDrop;
    result.cepstralDrop = cepstralDrop;
    result.energyRise = energyRise;
    result.region = regionToString (region);
    result.primaryRepresentation = primaryIndex == 0 ? "stft"
                                : primaryIndex == 1 ? "acf"
                                : primaryIndex == 2 ? "cepstrum"
                                : primaryIndex == 3 ? "cqt"
                                : "wavelet";
    result.recommendedWindow = chooseRecommendedWindow (region, adaptiveWindow);

    previousFrame.assign (samples, samples + numSamples);
    previousMagnitude = magnitude;
    previousACF = acf;
    previousCepstrum = cepstrum;
    previousFeatureFlux = transientBias;
    previousHarmonicRatio = harmonicRatio;
    previousACFPeriodicity = periodicity;
    previousCepstrumPeak = cepstralPeak;
    previousRms = rms;
    ++framesSeen;

    return result;
}
