#!/usr/bin/env python3
"""
Example usage script demonstrating Sarvam and Whisper integration.
Shows how to switch between providers and handle different configurations.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from transcriber_factory import UnifiedTranscriber


def example_sarvam_usage():
    """Example of using Sarvam provider."""
    print("=== Sarvam Provider Example ===")
    
    # Check if API key is set
    if not os.getenv('SARVAM_API_KEY'):
        print("⚠ SARVAM_API_KEY not set. Skipping Sarvam example.")
        print("Set it with: export SARVAM_API_KEY='your-api-key'")
        return False
    
    # Sarvam configuration
    sarvam_config = {
        'asr_provider': 'sarvam',
        'asr_model': 'saarika:v2.5',
        'asr_language_code': 'ta-IN',
        'asr_batch_enabled': True,
        'asr_concurrency_limit': 3,
        'transcript_cache_enabled': True,
        'temp_dir': './data/tmp'
    }
    
    try:
        transcriber = UnifiedTranscriber(sarvam_config)
        print("✓ Sarvam transcriber created successfully")
        print(f"  Provider: {transcriber.provider}")
        print(f"  Model: {transcriber.transcriber.model}")
        print(f"  Language: {transcriber.transcriber.language_code}")
        return True
    except Exception as e:
        print(f"✗ Failed to create Sarvam transcriber: {e}")
        return False


def example_whisper_usage():
    """Example of using Whisper provider."""
    print("\n=== Whisper Provider Example ===")
    
    # Whisper configuration
    whisper_config = {
        'asr_provider': 'whisper',
        'whisper_model': 'medium',
        'language': 'ta',
        'temp_dir': './data/tmp'
    }
    
    try:
        transcriber = UnifiedTranscriber(whisper_config)
        print("✓ Whisper transcriber created successfully")
        print(f"  Provider: {transcriber.provider}")
        print(f"  Model: {transcriber.transcriber.model_name}")
        print(f"  Language: {transcriber.transcriber.language}")
        return True
    except Exception as e:
        print(f"✗ Failed to create Whisper transcriber: {e}")
        return False


def example_config_loading():
    """Example of loading configuration from file."""
    print("\n=== Configuration Loading Example ===")
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration loaded from config.yaml")
        print(f"  ASR Provider: {config.get('asr_provider', 'sarvam')}")
        print(f"  ASR Model: {config.get('asr_model', 'saarika:v2.5')}")
        print(f"  Language Code: {config.get('asr_language_code', 'ta-IN')}")
        
        # Create transcriber from config
        transcriber = UnifiedTranscriber(config)
        print(f"✓ Transcriber created from config: {transcriber.provider}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def example_batch_processing():
    """Example of batch processing with different providers."""
    print("\n=== Batch Processing Example ===")
    
    # Find audio files
    audio_dir = Path("./audio")
    if not audio_dir.exists():
        print("✗ Audio directory not found")
        return False
    
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("✗ No audio files found")
        return False
    
    print(f"Found {len(audio_files)} audio files")
    
    # Test with Whisper (doesn't require API key)
    whisper_config = {
        'asr_provider': 'whisper',
        'whisper_model': 'medium',
        'language': 'ta',
        'temp_dir': './data/tmp'
    }
    
    try:
        transcriber = UnifiedTranscriber(whisper_config)
        print("✓ Whisper transcriber ready for batch processing")
        print("  Note: Whisper processes files individually (no true batch)")
        
        # For demonstration, show how batch would work
        if len(audio_files) > 0:
            print(f"  Would process: {audio_files[0].name}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create batch transcriber: {e}")
        return False


def example_provider_switching():
    """Example of switching between providers."""
    print("\n=== Provider Switching Example ===")
    
    # Base configuration
    base_config = {
        'temp_dir': './data/tmp',
        'asr_language_code': 'ta-IN',
        'language': 'ta'
    }
    
    providers = ['whisper', 'sarvam']
    
    for provider in providers:
        config = base_config.copy()
        config['asr_provider'] = provider
        
        if provider == 'sarvam':
            config.update({
                'asr_model': 'saarika:v2.5',
                'asr_batch_enabled': True,
                'asr_concurrency_limit': 3
            })
        else:  # whisper
            config.update({
                'whisper_model': 'medium'
            })
        
        try:
            transcriber = UnifiedTranscriber(config)
            print(f"✓ {provider.capitalize()} transcriber created successfully")
        except Exception as e:
            if provider == 'sarvam' and 'SARVAM_API_KEY' in str(e):
                print(f"⚠ {provider.capitalize()} transcriber failed (no API key)")
            else:
                print(f"✗ {provider.capitalize()} transcriber failed: {e}")
    
    return True


def main():
    """Run all examples."""
    print("Sarvam Integration Examples")
    print("=" * 40)
    
    examples = [
        example_config_loading,
        example_whisper_usage,
        example_sarvam_usage,
        example_batch_processing,
        example_provider_switching,
    ]
    
    passed = 0
    total = len(examples)
    
    for example in examples:
        try:
            if example():
                passed += 1
        except Exception as e:
            print(f"✗ Example {example.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Example Results: {passed}/{total} examples passed")
    
    if passed == total:
        print("✓ All examples passed!")
    else:
        print("⚠ Some examples failed (expected for Sarvam without API key)")
    
    print("\nNext steps:")
    print("1. Set SARVAM_API_KEY for Sarvam provider")
    print("2. Run: python src/main.py --config config.yaml")
    print("3. Or run: python src/main.py --config config.yaml --provider whisper")


if __name__ == "__main__":
    main()
