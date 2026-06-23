#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace
{
static constexpr int kRingBufferSize = 8192;
static constexpr int kAnalysisHopSamples = 256;
static constexpr int kWarmupFrames = 4;

static float dbToGain (float db)
{
    return std::pow (10.0f, db / 20.0f);
}
} // namespace

TransientCompassAudioProcessor::TransientCompassAudioProcessor()
    : AudioProcessor (juce::AudioProcessor::BusesProperties()
                      .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "Parameters", createParameterLayout())
{
    inputRingBuffer.resize (kRingBufferSize, 0.0f);
}

juce::AudioProcessorValueTreeState::ParameterLayout TransientCompassAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "threshold", "Sensitivity",
        juce::NormalisableRange<float> (0.05f, 0.95f, 0.001f, 0.7f),
        0.42f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "resetThreshold", "Reset Threshold",
        juce::NormalisableRange<float> (0.01f, 0.90f, 0.001f, 0.7f),
        0.15f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "shapeMs", "Shape Time",
        juce::NormalisableRange<float> (2.0f, 80.0f, 0.1f, 0.55f),
        18.0f,
        " ms"));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "attackBoostDb", "Attack Boost",
        juce::NormalisableRange<float> (0.0f, 18.0f, 0.01f, 0.55f),
        9.0f,
        " dB"));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "sustainCutDb", "Sustain Cut",
        juce::NormalisableRange<float> (0.0f, 18.0f, 0.01f, 0.55f),
        5.0f,
        " dB"));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "mix", "Mix",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f, 1.0f),
        1.0f));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "outputGainDb", "Output Gain",
        juce::NormalisableRange<float> (-12.0f, 12.0f, 0.01f, 0.55f),
        -1.5f,
        " dB"));

    layout.add (std::make_unique<juce::AudioParameterChoice> (
        "analysisWindow", "Analysis Window",
        juce::StringArray { "Auto", "512", "1024", "2048", "4096" },
        0));

    layout.add (std::make_unique<juce::AudioParameterBool> (
        "adaptiveWindow", "Adaptive Window", true));

    return layout;
}

void TransientCompassAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused (samplesPerBlock);
    currentSampleRate = sampleRate;
    analysisEngine.prepare (sampleRate, 4096);
    analysisEngine.reset();

    std::fill (inputRingBuffer.begin(), inputRingBuffer.end(), 0.0f);
    ringWriteIndex = 0;
    ringSamplesFilled = 0;
    samplesSinceLastAnalysis = 0;
    transientPulseSamplesRemaining = 0;
    transientPulseLengthSamples = juce::jmax (1, (int) std::round (*apvts.getRawParameterValue ("shapeMs") * currentSampleRate * 0.001));

    latestScore.store (0.0f);
    latestConfidence.store (0.0f);
    latestFlux.store (0.0f);
    latestAcfDrop.store (0.0f);
    latestCepstralDrop.store (0.0f);
    latestEnergyRise.store (0.0f);
    latestZ1.store (0.0f);
    latestZ2.store (0.0f);
    latestTransientPulse.store (0.0f);
    latestRegionIndex.store (0);
    latestRecommendedWindow.store (2048);
    latestAnalysisWindow.store (2048);
    latestPrimaryRepresentationIndex.store (0);
    latestEventCount.store (0);
    latestShaperDrive.store (0.0f);
}

void TransientCompassAudioProcessor::releaseResources()
{
}

bool TransientCompassAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
        return false;

    return true;
}

int TransientCompassAudioProcessor::choiceToWindowSize (int choiceIndex)
{
    switch (choiceIndex)
    {
        case 1: return 512;
        case 2: return 1024;
        case 3: return 2048;
        case 4: return 4096;
        default: return 2048;
    }
}

int TransientCompassAudioProcessor::regionStringToIndex (const juce::String& region)
{
    if (region == "periodic_harmonic") return 1;
    if (region == "noise_collapse") return 2;
    if (region == "transient_overloaded") return 3;
    if (region == "smooth_lowpass") return 4;
    return 0;
}

void TransientCompassAudioProcessor::analyseLatestFrame()
{
    const int analysisWindowChoice = (int) std::lround (*apvts.getRawParameterValue ("analysisWindow"));
    const bool adaptiveWindow = *apvts.getRawParameterValue ("adaptiveWindow") > 0.5f;

    int analysisWindow = choiceToWindowSize (analysisWindowChoice);
    if (analysisWindowChoice == 0)
        analysisWindow = adaptiveWindow ? latestRecommendedWindow.load() : 2048;

    analysisWindow = juce::jlimit (512, (int) inputRingBuffer.size(), analysisWindow);
    latestAnalysisWindow.store (analysisWindow);

    if (ringSamplesFilled < analysisWindow)
        return;

    std::vector<float> frame ((size_t) analysisWindow, 0.0f);
    const int startIndex = (ringWriteIndex - analysisWindow + (int) inputRingBuffer.size()) % (int) inputRingBuffer.size();
    for (int i = 0; i < analysisWindow; ++i)
        frame[(size_t) i] = inputRingBuffer[(size_t) ((startIndex + i) % (int) inputRingBuffer.size())];

    const float transientThreshold = *apvts.getRawParameterValue ("threshold");
    const float resetThreshold = *apvts.getRawParameterValue ("resetThreshold");
    const int refractoryFrames = 0;

    const auto result = analysisEngine.analyseFrame (frame.data(),
                                                     analysisWindow,
                                                     transientThreshold,
                                                     resetThreshold,
                                                     kWarmupFrames,
                                                     refractoryFrames,
                                                     adaptiveWindow);

    latestScore.store (result.score);
    latestConfidence.store (result.confidence);
    latestFlux.store (result.spectralFlux);
    latestAcfDrop.store (result.acfDrop);
    latestCepstralDrop.store (result.cepstralDrop);
    latestEnergyRise.store (result.energyRise);
    latestZ1.store (result.z1);
    latestZ2.store (result.z2);
    latestRecommendedWindow.store (result.recommendedWindow);
    latestRegionIndex.store (regionStringToIndex (result.region));
    latestPrimaryRepresentationIndex.store (result.primaryRepresentation == "stft" ? 0
                                           : result.primaryRepresentation == "acf" ? 1
                                           : result.primaryRepresentation == "cepstrum" ? 2
                                           : result.primaryRepresentation == "cqt" ? 3
                                           : 4);

    if (result.isTransient)
    {
        latestEventCount.fetch_add (1);
        transientPulseLengthSamples = juce::jmax (1, (int) std::round (*apvts.getRawParameterValue ("shapeMs") * currentSampleRate * 0.001));
        transientPulseSamplesRemaining = transientPulseLengthSamples;
        latestShaperDrive.store (juce::jlimit (0.0f, 1.0f, 0.35f + 0.65f * result.score));
        latestTransientPulse.store (latestShaperDrive.load());
    }
    else
    {
        latestShaperDrive.store (juce::jlimit (0.0f, 1.0f, latestShaperDrive.load() * 0.985f));
        latestTransientPulse.store (juce::jlimit (0.0f, 1.0f, latestTransientPulse.load() * 0.96f));
    }

    latestAnalysisWindow.store (analysisWindow);
}

void TransientCompassAudioProcessor::applyTransientShaping (juce::AudioBuffer<float>& buffer, int startSample, int numSamples)
{
    const float mix = *apvts.getRawParameterValue ("mix");
    const float attackBoostDb = *apvts.getRawParameterValue ("attackBoostDb");
    const float sustainCutDb = *apvts.getRawParameterValue ("sustainCutDb");
    const float outputGainDb = *apvts.getRawParameterValue ("outputGainDb");

    const float outputGain = dbToGain (outputGainDb);
    const float drive = latestShaperDrive.load();
    const int numChannels = buffer.getNumChannels();

    for (int i = 0; i < numSamples; ++i)
    {
        float gain = 1.0f;
        if (transientPulseSamplesRemaining > 0 && transientPulseLengthSamples > 0)
        {
            const float env = (float) transientPulseSamplesRemaining / (float) transientPulseLengthSamples;
            const float shapedDb = drive * (attackBoostDb * env - sustainCutDb * (1.0f - env));
            gain = dbToGain (shapedDb);
            --transientPulseSamplesRemaining;
        }

        const float wetGain = gain * outputGain;
        const float dryGain = outputGain;
        for (int channel = 0; channel < numChannels; ++channel)
        {
            auto* channelData = buffer.getWritePointer (channel);
            const float dry = channelData[(size_t) startSample + (size_t) i];
            const float wet = dry * wetGain;
            channelData[(size_t) startSample + (size_t) i] = juce::jlimit (-2.0f, 2.0f, dry * (1.0f - mix) * dryGain + wet * mix);
        }
    }
}

void TransientCompassAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);
    juce::ScopedNoDenormals noDenormals;
    const auto totalNumInputChannels = getTotalNumInputChannels();
    const auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto channel = totalNumInputChannels; channel < totalNumOutputChannels; ++channel)
        buffer.clear (channel, 0, buffer.getNumSamples());

    const int numSamples = buffer.getNumSamples();
    if (totalNumInputChannels == 0 || numSamples <= 0)
        return;

    if (transientPulseLengthSamples <= 0)
        transientPulseLengthSamples = juce::jmax (1, (int) std::round (*apvts.getRawParameterValue ("shapeMs") * currentSampleRate * 0.001));

    for (int sample = 0; sample < numSamples; ++sample)
    {
        float mix = 0.0f;
        for (int channel = 0; channel < totalNumInputChannels; ++channel)
            mix += buffer.getReadPointer (channel)[sample];
        mix /= (float) totalNumInputChannels;

        inputRingBuffer[(size_t) ringWriteIndex] = mix;
        ringWriteIndex = (ringWriteIndex + 1) % (int) inputRingBuffer.size();
        ringSamplesFilled = juce::jmin ((int) inputRingBuffer.size(), ringSamplesFilled + 1);
    }

    samplesSinceLastAnalysis += numSamples;

    while (samplesSinceLastAnalysis >= kAnalysisHopSamples)
    {
        samplesSinceLastAnalysis -= kAnalysisHopSamples;
        analyseLatestFrame();
    }

    applyTransientShaping (buffer, 0, numSamples);
}

void TransientCompassAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto stateXml = apvts.state.createXml())
        copyXmlToBinary (*stateXml, destData);
}

void TransientCompassAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xmlState = getXmlFromBinary (data, sizeInBytes))
    {
        if (xmlState->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
    }
}

juce::AudioProcessorEditor* TransientCompassAudioProcessor::createEditor()
{
    return new TransientCompassAudioProcessorEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new TransientCompassAudioProcessor();
}
