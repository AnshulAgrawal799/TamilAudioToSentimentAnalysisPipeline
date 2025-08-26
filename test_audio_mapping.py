#!/usr/bin/env python3
"""
Test script to verify audio mapping functionality.
Tests the new audio_file_id, start_ms, end_ms, and duration_ms fields.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analyze import Segment, NLUAnalyzer
from transcribe import TranscriptionResult
from aggregate import Aggregator

def test_segment_audio_mapping():
    """Test that segments include audio mapping fields."""
    print("Testing Segment audio mapping fields...")
    
    # Create a test segment with audio mapping
    segment = Segment(
        seller_id="S123",
        stop_id="STOP45",
        audio_file_id="tmp/Audio1.wav",
        start_ms=5220,
        end_ms=7240,
        duration_ms=2020,
        textTamil="தக்காளி எவ்வளவு விலை",
        textEnglish="What is the price of tomatoes?"
    )
    
    # Convert to dict and verify fields
    segment_dict = segment.to_dict()
    
    required_fields = ['audio_file_id', 'start_ms', 'end_ms', 'duration_ms']
    for field in required_fields:
        if field not in segment_dict:
            print(f"❌ Missing field: {field}")
            return False
        print(f"✅ Found field: {field} = {segment_dict[field]}")
    
    # Verify timing calculations
    if segment_dict['duration_ms'] != (segment_dict['end_ms'] - segment_dict['start_ms']):
        print("❌ Duration calculation incorrect")
        return False
    
    print("✅ Duration calculation correct")
    return True

def test_nlu_analyzer_audio_mapping():
    """Test that NLUAnalyzer creates segments with audio mapping."""
    print("\nTesting NLUAnalyzer audio mapping...")
    
    # Create mock transcription result
    class MockTranscriptionResult:
        def __init__(self):
            self.audio_file = Path("tmp/Audio1.wav")
            self.metadata = {
                'seller_id': 'S123',
                'stop_id': 'STOP45',
                'recording_start': '2025-08-19T00:00:00Z'
            }
        
        def get_segments(self):
            return [
                {'start': 5.22, 'end': 7.24, 'text': 'தக்காளி எவ்வளவு விலை'},
                {'start': 11.32, 'end': 13.45, 'text': 'ரூபாய் இருபது கிலோ'}
            ]
    
    # Create analyzer with minimal config
    config = {
        'placeholders': {
            'seller_id': 'S123',
            'stop_id': 'STOP45',
            'anchor_time': '2025-08-19T00:00:00Z'
        },
        'confidence_threshold': 0.3,
        'min_segment_words': 2
    }
    
    analyzer = NLUAnalyzer(config)
    mock_result = MockTranscriptionResult()
    
    # Analyze and check segments
    segments = analyzer.analyze(mock_result)
    
    if len(segments) != 2:
        print(f"❌ Expected 2 segments, got {len(segments)}")
        return False
    
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}:")
        print(f"  audio_file_id: {segment.audio_file_id}")
        print(f"  start_ms: {segment.start_ms}")
        print(f"  end_ms: {segment.end_ms}")
        print(f"  duration_ms: {segment.duration_ms}")
        
        # Verify timing fields are present and reasonable
        if not all([segment.audio_file_id, segment.start_ms, segment.end_ms, segment.duration_ms]):
            print(f"❌ Segment {i+1} missing timing fields")
            return False
        
        if segment.duration_ms != (segment.end_ms - segment.start_ms):
            print(f"❌ Segment {i+1} duration calculation incorrect")
            return False
    
    print("✅ All segments have correct audio mapping fields")
    return True

def test_aggregator_audio_stats():
    """Test that Aggregator includes audio file statistics."""
    print("\nTesting Aggregator audio file statistics...")
    
    # Create test segments
    segments = [
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            audio_file_id="tmp/Audio1.wav",
            start_ms=1000,
            end_ms=3000,
            duration_ms=2000,
            textTamil="Test text 1"
        ),
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            audio_file_id="tmp/Audio1.wav",
            start_ms=4000,
            end_ms=6000,
            duration_ms=2000,
            textTamil="Test text 2"
        ),
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            audio_file_id="tmp/Audio2.wav",
            start_ms=0,
            end_ms=2000,
            duration_ms=2000,
            textTamil="Test text 3"
        )
    ]
    
    # Create aggregator
    config = {'placeholders': {}}
    aggregator = Aggregator(config)
    
    # Test stop-level aggregation
    stop_aggregates = aggregator.aggregate_by_stop(segments)
    
    if 'STOP45' not in stop_aggregates:
        print("❌ Stop aggregate not found")
        return False
    
    stop_agg = stop_aggregates['STOP45']
    
    if 'audio_files' not in stop_agg:
        print("❌ audio_files field missing from stop aggregate")
        return False
    
    audio_files = stop_agg['audio_files']
    print(f"Audio files in stop aggregate: {list(audio_files.keys())}")
    
    # Verify Audio1.wav stats
    if 'tmp/Audio1.wav' in audio_files:
        audio1_stats = audio_files['tmp/Audio1.wav']
        print(f"Audio1 stats: {audio1_stats}")
        
        if audio1_stats['n_segments'] != 2:
            print("❌ Audio1 segment count incorrect")
            return False
        
        if audio1_stats['total_duration_ms'] != 4000:
            print("❌ Audio1 total duration incorrect")
            return False
    
    # Verify Audio2.wav stats
    if 'tmp/Audio2.wav' in audio_files:
        audio2_stats = audio_files['tmp/Audio2.wav']
        print(f"Audio2 stats: {audio2_stats}")
        
        if audio2_stats['n_segments'] != 1:
            print("❌ Audio2 segment count incorrect")
            return False
    
    print("✅ Stop-level audio file statistics correct")
    
    # Test day-level aggregation
    day_aggregates = aggregator.aggregate_by_day(segments)
    
    if 'S123' not in day_aggregates:
        print("❌ Day aggregate not found")
        return False
    
    day_agg = day_aggregates['S123']
    
    if 'audio_files' not in day_agg:
        print("❌ audio_files field missing from day aggregate")
        return False
    
    print("✅ Day-level audio file statistics present")
    return True

def main():
    """Run all audio mapping tests."""
    print("🧪 Testing Audio Mapping Functionality\n")
    
    tests = [
        test_segment_audio_mapping,
        test_nlu_analyzer_audio_mapping,
        test_aggregator_audio_stats
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All audio mapping tests passed!")
        return True
    else:
        print("❌ Some audio mapping tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
