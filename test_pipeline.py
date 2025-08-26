#!/usr/bin/env python3
"""
Simple test script to validate pipeline components.
Run this to check if all modules can be imported and basic functionality works.
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from ingest_audio import AudioIngester
        logger.info("✓ ingest_audio imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import ingest_audio: {e}")
        return False
    
    try:
        from transcribe import Transcriber, TranscriptionResult
        logger.info("✓ transcribe imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import transcribe: {e}")
        return False
    
    try:
        from analyze import NLUAnalyzer, Segment
        logger.info("✓ analyze imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import analyze: {e}")
        return False
    
    try:
        from aggregate import Aggregator
        logger.info("✓ aggregate imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import aggregate: {e}")
        return False
    
    try:
        from main import load_config, setup_directories
        logger.info("✓ main imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import main: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['audio_dir', 'output_dir', 'temp_dir', 'placeholders']
        for key in required_keys:
            if key not in config:
                logger.error(f"✗ Missing required config key: {key}")
                return False
        
        logger.info("✓ Configuration loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return False

def test_segment_creation():
    """Test Segment object creation and serialization."""
    logger.info("Testing Segment object functionality...")
    
    try:
        from analyze import Segment
        
        # Create a test segment with new audio mapping fields
        segment = Segment(
            seller_id="TEST123",
            stop_id="STOP_TEST",
            audio_file_id="tmp/Audio1.wav",
            start_ms=5220,
            end_ms=7240,
            duration_ms=2020,
            textTamil="Test Tamil text",
            textEnglish="Test English text"
        )
        
        # Test dictionary conversion
        segment_dict = segment.to_dict()
        if not isinstance(segment_dict, dict):
            logger.error("✗ Segment.to_dict() did not return a dictionary")
            return False
        
        # Test that new fields are present
        required_fields = ['audio_file_id', 'start_ms', 'end_ms', 'duration_ms']
        for field in required_fields:
            if field not in segment_dict:
                logger.error(f"✗ Missing required field: {field}")
                return False
        
        # Test JSON serialization
        segment_json = segment.to_json()
        if not isinstance(segment_json, str):
            logger.error("✗ Segment.to_json() did not return a string")
            return False
        
        logger.info("✓ Segment object functionality works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to test Segment functionality: {e}")
        return False

def test_analyzer_creation():
    """Test NLUAnalyzer creation."""
    logger.info("Testing NLUAnalyzer creation...")
    
    try:
        from analyze import NLUAnalyzer
        
        # Load config
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create analyzer
        analyzer = NLUAnalyzer(config)
        
        if not hasattr(analyzer, 'placeholders'):
            logger.error("✗ NLUAnalyzer missing placeholders attribute")
            return False
        
        logger.info("✓ NLUAnalyzer created successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create NLUAnalyzer: {e}")
        return False

def test_aggregator_creation():
    """Test Aggregator creation."""
    logger.info("Testing Aggregator creation...")
    
    try:
        from aggregate import Aggregator
        
        # Load config
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create aggregator
        aggregator = Aggregator(config)
        
        if not hasattr(aggregator, 'placeholders'):
            logger.error("✗ Aggregator missing placeholders attribute")
            return False
        
        logger.info("✓ Aggregator created successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create Aggregator: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting pipeline validation tests...")
    
    tests = [
        test_imports,
        test_config_loading,
        test_segment_creation,
        test_analyzer_creation,
        test_aggregator_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("")  # Empty line for readability
    
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pipeline is ready to use.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
