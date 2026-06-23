cmake_minimum_required(VERSION 3.22)

project(TransientCompass VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (DEFINED ENV{JUCE_DIR})
    set(JUCE_DIR $ENV{JUCE_DIR})
else()
    set(JUCE_DIR "/Users/user/Desktop/wavsynth/JUCE")
endif()

if (NOT EXISTS ${JUCE_DIR})
    message(FATAL_ERROR "JUCE directory not found. Set JUCE_DIR to a local JUCE checkout.")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

juce_add_plugin(TransientCompass
    COMPANY_NAME "SpacekidLabs"
    PLUGIN_MANUFACTURER_CODE SKid
    PLUGIN_CODE TrCm
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "TransientCompass"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
)

target_sources(TransientCompass PRIVATE
    Source/PluginProcessor.h
    Source/PluginProcessor.cpp
    Source/PluginEditor.h
    Source/PluginEditor.cpp
    Source/CompassAnalysis.h
    Source/CompassAnalysis.cpp
)

target_compile_definitions(TransientCompass PRIVATE
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_compile_features(TransientCompass PRIVATE cxx_std_20)

target_link_libraries(TransientCompass PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_dsp
    juce::juce_gui_extra
    juce::juce_gui_basics
    juce::juce_graphics
    juce::juce_events
    juce::juce_core
)

juce_generate_juce_header(TransientCompass)
