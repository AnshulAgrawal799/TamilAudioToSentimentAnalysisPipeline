#!/usr/bin/env python3
"""
Test script for Sarvam integration.
Tests both Sarvam and Whisper providers to ensure they work correctly.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from transcriber_factory import UnifiedTranscriber


def test_provider_creation():
    """Test that both providers can be created successfully."""
    print("Testing provider creation...")
    
    # Test configs
    sarvam_config = {
        'asr_provider': 'sarvam',
        'asr_model': 'saarika:v2.5',
        'asr_language_code': 'ta-IN',
        'temp_dir': './data/tmp'
    }
    
    whisper_config = {
        'asr_provider': 'whisper',
        'whisper_model': 'medium',
        'language': 'ta',
        'temp_dir': './data/tmp'
    }
    
    try:
        # Test Sarvam transcriber creation (will fail without API key, but should not crash)
        sarvam_transcriber = UnifiedTranscriber(sarvam_config)
        print("✓ Sarvam transcriber created successfully")
    except ValueError as e:
        if "SARVAM_API_KEY" in str(e):
            print("✓ Sarvam transcriber creation failed as expected (no API key)")
        else:
            print(f"✗ Sarvam transcriber creation failed unexpectedly: {e}")
            return False
    
    try:
        # Test Whisper transcriber creation
        whisper_transcriber = UnifiedTranscriber(whisper_config)
        print("✓ Whisper transcriber created successfully")
    except Exception as e:
        print(f"✗ Whisper transcriber creation failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test that the config file can be loaded and parsed correctly."""
    print("\nTesting config loading...")
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['asr_provider', 'asr_model', 'asr_language_code']
        for field in required_fields:
            if field not in config:
                print(f"✗ Missing required field: {field}")
                return False
        
        print("✓ Config loaded successfully")
        print(f"  ASR Provider: {config['asr_provider']}")
        print(f"  ASR Model: {config['asr_model']}")
        print(f"  Language Code: {config['asr_language_code']}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_audio_file_detection():
    """Test that audio files can be detected."""
    print("\nTesting audio file detection...")
    
    audio_dir = Path("./audio")
    if not audio_dir.exists():
        print("✗ Audio directory not found")
        return False
    
    wav_files = list(audio_dir.glob("*.wav"))
    mp3_files = list(audio_dir.glob("*.mp3"))
    
    print(f"  Found {len(wav_files)} WAV files")
    print(f"  Found {len(mp3_files)} MP3 files")
    
    if len(wav_files) == 0 and len(mp3_files) == 0:
        print("✗ No audio files found")
        return False
    
    print("✓ Audio files detected")
    return True


def test_environment_setup():
    """Test environment setup for Sarvam."""
    print("\nTesting environment setup...")
    
    # Check if SARVAM_API_KEY is set
    api_key = os.getenv('SARVAM_API_KEY')
    if api_key:
        print("✓ SARVAM_API_KEY is set")
        print(f"  Key length: {len(api_key)} characters")
    else:
        print("⚠ SARVAM_API_KEY is not set (required for Sarvam provider)")
        print("  Set it with: export SARVAM_API_KEY='your-api-key'")
    
    # Check Python dependencies
    try:
        import requests
        print("✓ requests library available")
    except ImportError:
        print("✗ requests library not available")
        return False
    
    try:
        import whisper
        print("✓ whisper library available")
    except ImportError:
        print("✗ whisper library not available")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Sarvam Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_config_loading,
        test_environment_setup,
        test_provider_creation,
        test_audio_file_detection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Integration is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
