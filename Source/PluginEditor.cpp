#include "PluginEditor.h"

namespace
{
static juce::Colour regionColourForIndex (int index)
{
    switch (index)
    {
        case 1: return juce::Colour::fromString ("#2EA44F");
        case 2: return juce::Colour::fromString ("#F85149");
        case 3: return juce::Colour::fromString ("#58A6FF");
        case 4: return juce::Colour::fromString ("#BC8CFF");
        default: return juce::Colour::fromString ("#8B949E");
    }
}

static juce::String regionTextForIndex (int index)
{
    switch (index)
    {
        case 1: return "Periodic Harmonic";
        case 2: return "Noise Collapse";
        case 3: return "Transient Overloaded";
        case 4: return "Smooth Lowpass";
        default: return "Transition Zone";
    }
}

static juce::String representationTextForIndex (int index)
{
    switch (index)
    {
        case 0: return "STFT";
        case 1: return "ACF";
        case 2: return "Cepstrum";
        case 3: return "CQT";
        case 4: return "Wavelet";
        default: return "Unknown";
    }
}
} // namespace

CompassScopeComponent::CompassScopeComponent (TransientCompassAudioProcessor& p)
    : processor (p)
{
}

void CompassScopeComponent::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat();
    g.setColour (juce::Colour::fromString ("#0D1117"));
    g.fillRoundedRectangle (bounds, 12.0f);

    g.setColour (juce::Colour::fromString ("#30363D"));
    g.drawRoundedRectangle (bounds, 12.0f, 1.2f);

    const float margin = 14.0f;
    const float w = bounds.getWidth();
    const float h = bounds.getHeight();

    auto mapPoint = [=] (float z1, float z2)
    {
        const float x = margin + ((z1 + 3.0f) / 9.0f) * (w - 2.0f * margin);
        const float y = h - margin - ((z2 + 3.0f) / 6.0f) * (h - 2.0f * margin);
        return juce::Point<float> (x, y);
    };

    g.setColour (juce::Colour::fromString ("#2EA44F").withAlpha (0.08f));
    auto harmonicTopLeft = mapPoint (-3.0f, 3.0f);
    auto harmonicBottomRight = mapPoint (-0.5f, -0.2f);
    g.fillRect (harmonicTopLeft.x,
                harmonicBottomRight.y,
                harmonicBottomRight.x - harmonicTopLeft.x,
                harmonicTopLeft.y - harmonicBottomRight.y);

    g.setColour (juce::Colour::fromString ("#F85149").withAlpha (0.08f));
    auto noiseTopLeft = mapPoint (1.5f, 3.0f);
    auto noiseBottomRight = mapPoint (6.0f, -3.0f);
    g.fillRect (noiseTopLeft.x,
                noiseBottomRight.y,
                noiseBottomRight.x - noiseTopLeft.x,
                noiseTopLeft.y - noiseBottomRight.y);

    g.setColour (juce::Colour::fromString ("#58A6FF").withAlpha (0.08f));
    auto transientTopLeft = mapPoint (-3.0f, 3.0f);
    auto transientBottomRight = mapPoint (1.5f, 1.5f);
    g.fillRect (transientTopLeft.x,
                transientBottomRight.y,
                transientBottomRight.x - transientTopLeft.x,
                transientTopLeft.y - transientBottomRight.y);

    g.setColour (juce::Colour::fromString ("#BC8CFF").withAlpha (0.08f));
    auto lowpassTopLeft = mapPoint (-0.5f, -0.2f);
    auto lowpassBottomRight = mapPoint (1.5f, -3.0f);
    g.fillRect (lowpassTopLeft.x,
                lowpassBottomRight.y,
                lowpassBottomRight.x - lowpassTopLeft.x,
                lowpassTopLeft.y - lowpassBottomRight.y);

    auto origin = mapPoint (0.0f, 0.0f);
    g.setColour (juce::Colour::fromString ("#21262D"));
    g.drawHorizontalLine ((int) origin.y, margin, w - margin);
    g.drawVerticalLine ((int) origin.x, margin, h - margin);

    g.setFont (juce::Font (12.0f, juce::Font::bold));
    g.setColour (juce::Colours::white.withAlpha (0.25f));
    g.drawText ("Transient Compass", 16, 12, 150, 18, juce::Justification::left, false);

    const auto z1 = processor.latestZ1.load();
    const auto z2 = processor.latestZ2.load();
    const auto dot = mapPoint (juce::jlimit (-3.0f, 6.0f, z1), juce::jlimit (-3.0f, 3.0f, z2));
    const float pulse = processor.latestTransientPulse.load();

    g.setColour (juce::Colours::orange.withAlpha (0.22f + 0.45f * pulse));
    g.fillEllipse (dot.x - 12.0f, dot.y - 12.0f, 24.0f, 24.0f);
    g.setColour (juce::Colours::orange);
    g.fillEllipse (dot.x - 4.5f, dot.y - 4.5f, 9.0f, 9.0f);

    g.setColour (juce::Colours::white.withAlpha (0.55f));
    g.setFont (10.0f);
    g.drawText ("z1", (int) dot.x + 8, (int) dot.y - 18, 20, 12, juce::Justification::left, false);
    g.drawText ("z2", (int) dot.x + 8, (int) dot.y - 6, 20, 12, juce::Justification::left, false);
}

TransientCompassAudioProcessorEditor::TransientCompassAudioProcessorEditor (TransientCompassAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p), scope (p)
{
    configureSlider (thresholdSlider, juce::Slider::LinearHorizontal);
    configureSlider (resetSlider, juce::Slider::LinearHorizontal);
    configureSlider (shapeSlider, juce::Slider::LinearHorizontal);
    configureSlider (attackSlider, juce::Slider::LinearHorizontal);
    configureSlider (sustainSlider, juce::Slider::LinearHorizontal);
    configureSlider (mixSlider, juce::Slider::LinearHorizontal);
    configureSlider (outputGainSlider, juce::Slider::LinearHorizontal);

    thresholdSlider.setTextValueSuffix ("");
    resetSlider.setTextValueSuffix ("");
    shapeSlider.setTextValueSuffix (" ms");
    attackSlider.setTextValueSuffix (" dB");
    sustainSlider.setTextValueSuffix (" dB");
    mixSlider.setTextValueSuffix ("");
    outputGainSlider.setTextValueSuffix (" dB");

    adaptiveWindowButton.setButtonText ("Adaptive window");
    addAndMakeVisible (adaptiveWindowButton);

    windowSelector.addItem ("Auto", 1);
    windowSelector.addItem ("512", 2);
    windowSelector.addItem ("1024", 3);
    windowSelector.addItem ("2048", 4);
    windowSelector.addItem ("4096", 5);

    addAndMakeVisible (scope);
    addAndMakeVisible (thresholdSlider);
    addAndMakeVisible (resetSlider);
    addAndMakeVisible (shapeSlider);
    addAndMakeVisible (attackSlider);
    addAndMakeVisible (sustainSlider);
    addAndMakeVisible (mixSlider);
    addAndMakeVisible (outputGainSlider);
    addAndMakeVisible (windowSelector);
    addAndMakeVisible (adaptiveWindowButton);

    for (auto* label : { &scoreLabel, &confidenceLabel, &regionLabel, &representationLabel, &windowLabel, &countLabel })
    {
        label->setJustificationType (juce::Justification::left);
        label->setColour (juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible (label);
    }

    thresholdAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (processor.apvts, "threshold", thresholdSlider);
    resetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (processor.apvts, "resetThreshold", resetSlider);
    shapeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (processor.apvts, "shapeMs", shapeSlider);
    attackAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (processor.apvts, "attackBoostDb", attackSlider);
    sustainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (processor.apvts, "sustainCutDb", sustainSlider);
    mixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (processor.apvts, "mix", mixSlider);
    outputGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (processor.apvts, "outputGainDb", outputGainSlider);
    windowAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (processor.apvts, "analysisWindow", windowSelector);
    adaptiveWindowAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (processor.apvts, "adaptiveWindow", adaptiveWindowButton);

    setSize (980, 720);
    startTimerHz (30);
    updateStatusTexts();
}

TransientCompassAudioProcessorEditor::~TransientCompassAudioProcessorEditor()
{
    stopTimer();
}

void TransientCompassAudioProcessorEditor::configureSlider (juce::Slider& slider, juce::Slider::SliderStyle style)
{
    slider.setSliderStyle (style);
    slider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 64, 18);
    slider.setColour (juce::Slider::thumbColourId, juce::Colour::fromString ("#58A6FF"));
    slider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#30363D"));
    slider.setColour (juce::Slider::rotarySliderFillColourId, juce::Colour::fromString ("#58A6FF"));
    slider.setColour (juce::Slider::textBoxTextColourId, juce::Colours::white);
    addAndMakeVisible (slider);
}

void TransientCompassAudioProcessorEditor::timerCallback()
{
    const auto pulse = processor.latestTransientPulse.load();
    if (pulse > 0.001f)
        processor.latestTransientPulse.store (pulse * 0.92f);

    updateStatusTexts();
    repaint();
}

void TransientCompassAudioProcessorEditor::updateStatusTexts()
{
    scoreLabel.setText ("Score: " + juce::String (processor.latestScore.load(), 3), juce::dontSendNotification);
    confidenceLabel.setText ("Confidence: " + juce::String (processor.latestConfidence.load(), 3), juce::dontSendNotification);
    regionLabel.setText ("Region: " + regionTextForIndex (processor.latestRegionIndex.load()), juce::dontSendNotification);
    representationLabel.setText ("Primary: " + representationTextForIndex (processor.latestPrimaryRepresentationIndex.load()), juce::dontSendNotification);
    windowLabel.setText ("Recommended window: " + juce::String (processor.latestRecommendedWindow.load()) + " samples", juce::dontSendNotification);
    countLabel.setText ("Events: " + juce::String (processor.latestEventCount.load()), juce::dontSendNotification);
}

void TransientCompassAudioProcessorEditor::paint (juce::Graphics& g)
{
    juce::ColourGradient gradient (juce::Colour::fromString ("#161B22"), 0.0f, 0.0f,
                                   juce::Colour::fromString ("#0D1117"), 0.0f, (float) getHeight(), false);
    g.setGradientFill (gradient);
    g.fillAll();

    g.setColour (juce::Colours::white);
    g.setFont (juce::Font (18.0f, juce::Font::bold));
    g.drawText ("Transient Compass", 24, 18, 280, 24, juce::Justification::left, true);

    g.setColour (juce::Colours::white.withAlpha (0.55f));
    g.setFont (juce::Font (11.0f, juce::Font::plain));
    g.drawText ("Transient shaper driven by the same framework analysis", 24, 42, 420, 18, juce::Justification::left, true);

    const auto score = processor.latestScore.load();
    const auto confidence = processor.latestConfidence.load();
    const auto pulse = processor.latestTransientPulse.load();
    const auto regionIndex = processor.latestRegionIndex.load();
    const int controlLabelX = 468;

    g.setColour (juce::Colour::fromString ("#30363D"));
    g.fillRoundedRectangle (18.0f, 76.0f, 400.0f, 96.0f, 12.0f);
    g.setColour (juce::Colour::fromString ("#21262D"));
    g.drawRoundedRectangle (18.0f, 76.0f, 400.0f, 96.0f, 12.0f, 1.1f);

    g.setColour (juce::Colours::white.withAlpha (0.7f));
    g.setFont (juce::Font (13.0f, juce::Font::bold));
    g.drawText ("Live response", 34, 86, 120, 18, juce::Justification::left, true);

    g.setColour (juce::Colours::white.withAlpha (0.18f));
    g.fillRoundedRectangle (34.0f, 110.0f, 360.0f, 14.0f, 7.0f);
    g.setColour (juce::Colour::fromString ("#58A6FF").withAlpha (0.85f));
    g.fillRoundedRectangle (34.0f, 110.0f, 360.0f * juce::jlimit (0.0f, 1.0f, score), 14.0f, 7.0f);

    g.setColour (juce::Colours::white.withAlpha (0.7f));
    g.setFont (10.5f);
    g.drawText ("Score", 34, 128, 50, 16, juce::Justification::left, true);
    g.drawText ("Confidence", 120, 128, 70, 16, juce::Justification::left, true);

    g.setColour (juce::Colours::white);
    g.drawText (juce::String (score, 3), 72, 128, 38, 16, juce::Justification::left, true);
    g.drawText (juce::String (confidence, 3), 198, 128, 38, 16, juce::Justification::left, true);

    g.setColour (regionColourForIndex (regionIndex).withAlpha (0.9f));
    g.fillEllipse (356.0f, 82.0f, 44.0f, 44.0f);
    g.setColour (juce::Colours::white);
    g.setFont (10.0f);
    g.drawFittedText (regionTextForIndex (regionIndex), 308, 130, 140, 18, juce::Justification::centred, 1);

    if (pulse > 0.02f)
    {
        g.setColour (juce::Colours::orange.withAlpha (0.12f + 0.22f * pulse));
        g.fillRoundedRectangle (24.0f, 172.0f, 388.0f, 8.0f, 4.0f);
    }

    g.setColour (juce::Colours::white.withAlpha (0.5f));
    g.setFont (10.0f);
    g.drawText ("The detector now shapes attack and sustain so you can hear it directly on drums.", 24, 186, 460, 18, juce::Justification::left, true);

    g.setColour (juce::Colours::white.withAlpha (0.7f));
    g.setFont (10.0f);
    g.drawText ("Sensitivity", controlLabelX, 74, 120, 14, juce::Justification::left, true);
    g.drawText ("Reset floor", controlLabelX, 124, 120, 14, juce::Justification::left, true);
    g.drawText ("Shape time", controlLabelX, 174, 120, 14, juce::Justification::left, true);
    g.drawText ("Attack boost", controlLabelX, 224, 120, 14, juce::Justification::left, true);
    g.drawText ("Sustain cut", controlLabelX, 274, 120, 14, juce::Justification::left, true);
    g.drawText ("Mix", controlLabelX, 324, 120, 14, juce::Justification::left, true);
    g.drawText ("Output gain", controlLabelX, 374, 120, 14, juce::Justification::left, true);
    g.drawText ("Analysis window", controlLabelX, 424, 120, 14, juce::Justification::left, true);
    g.drawText ("Adaptive window", controlLabelX, 468, 120, 14, juce::Justification::left, true);
}

void TransientCompassAudioProcessorEditor::resized()
{
    scope.setBounds (18, 220, 420, 370);

    const int rightX = 468;
    const int rowW = 232;
    const int controlH = 28;

    thresholdSlider.setBounds (rightX, 92, rowW, controlH);
    resetSlider.setBounds (rightX, 142, rowW, controlH);
    shapeSlider.setBounds (rightX, 192, rowW, controlH);
    attackSlider.setBounds (rightX, 242, rowW, controlH);
    sustainSlider.setBounds (rightX, 292, rowW, controlH);
    mixSlider.setBounds (rightX, 342, rowW, controlH);
    outputGainSlider.setBounds (rightX, 392, rowW, controlH);

    windowSelector.setBounds (rightX, 442, rowW, 28);
    adaptiveWindowButton.setBounds (rightX, 486, rowW, 26);

    scoreLabel.setBounds (rightX, 530, rowW, 20);
    confidenceLabel.setBounds (rightX, 554, rowW, 20);
    regionLabel.setBounds (rightX, 578, rowW, 20);
    representationLabel.setBounds (rightX, 602, rowW, 20);
    windowLabel.setBounds (rightX, 626, rowW, 20);
    countLabel.setBounds (rightX, 650, rowW, 20);
}
