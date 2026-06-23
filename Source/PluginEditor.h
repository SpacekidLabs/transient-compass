#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class CompassScopeComponent  : public juce::Component
{
public:
    explicit CompassScopeComponent (TransientCompassAudioProcessor&);
    void paint (juce::Graphics& g) override;

private:
    TransientCompassAudioProcessor& processor;
};

class TransientCompassAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                              private juce::Timer
{
public:
    explicit TransientCompassAudioProcessorEditor (TransientCompassAudioProcessor&);
    ~TransientCompassAudioProcessorEditor() override;

    void paint (juce::Graphics& g) override;
    void resized() override;

private:
    void timerCallback() override;
    void configureSlider (juce::Slider& slider, juce::Slider::SliderStyle style);
    void updateStatusTexts();

    TransientCompassAudioProcessor& processor;
    CompassScopeComponent scope;

    juce::Slider thresholdSlider;
    juce::Slider resetSlider;
    juce::Slider shapeSlider;
    juce::Slider attackSlider;
    juce::Slider sustainSlider;
    juce::Slider mixSlider;
    juce::Slider outputGainSlider;
    juce::ComboBox windowSelector;
    juce::ToggleButton adaptiveWindowButton;

    juce::Label scoreLabel;
    juce::Label confidenceLabel;
    juce::Label regionLabel;
    juce::Label representationLabel;
    juce::Label windowLabel;
    juce::Label countLabel;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> thresholdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> resetAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> shapeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attackAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> sustainAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> outputGainAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> windowAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> adaptiveWindowAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TransientCompassAudioProcessorEditor)
};
