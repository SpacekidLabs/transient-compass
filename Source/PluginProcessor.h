#pragma once

#include <JuceHeader.h>
#include "CompassAnalysis.h"
#include <atomic>
#include <vector>

class TransientCompassAudioProcessor  : public juce::AudioProcessor
{
public:
    TransientCompassAudioProcessor();
    ~TransientCompassAudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "TransientCompass"; }

    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int index) override { juce::ignoreUnused (index); }
    const juce::String getProgramName (int index) override { juce::ignoreUnused (index); return {}; }
    void changeProgramName (int index, const juce::String& newName) override
    {
        juce::ignoreUnused (index, newName);
    }

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

    std::atomic<float> latestScore { 0.0f };
    std::atomic<float> latestConfidence { 0.0f };
    std::atomic<float> latestFlux { 0.0f };
    std::atomic<float> latestAcfDrop { 0.0f };
    std::atomic<float> latestCepstralDrop { 0.0f };
    std::atomic<float> latestEnergyRise { 0.0f };
    std::atomic<float> latestZ1 { 0.0f };
    std::atomic<float> latestZ2 { 0.0f };
    std::atomic<float> latestTransientPulse { 0.0f };
    std::atomic<int> latestRegionIndex { 0 };
    std::atomic<int> latestRecommendedWindow { 2048 };
    std::atomic<int> latestAnalysisWindow { 2048 };
    std::atomic<int> latestPrimaryRepresentationIndex { 0 };
    std::atomic<int> latestEventCount { 0 };
    std::atomic<float> latestShaperDrive { 0.0f };

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    static int choiceToWindowSize (int choiceIndex);
    static int regionStringToIndex (const juce::String& region);

    void analyseLatestFrame();
    void applyTransientShaping (juce::AudioBuffer<float>& buffer, int startSample, int numSamples);

    CompassAnalysisEngine analysisEngine;
    std::vector<float> inputRingBuffer;
    int ringWriteIndex = 0;
    int ringSamplesFilled = 0;
    int samplesSinceLastAnalysis = 0;
    int transientPulseSamplesRemaining = 0;
    int transientPulseLengthSamples = 768;
    double currentSampleRate = 48000.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TransientCompassAudioProcessor)
};
