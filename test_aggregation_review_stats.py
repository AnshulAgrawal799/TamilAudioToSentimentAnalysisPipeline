#!/usr/bin/env python3
"""
Tests for review statistics in aggregation outputs.
"""

import sys
import os
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyze import Segment
from aggregate import Aggregator


def expect(condition: bool, message: str) -> bool:
    if not condition:
        print(f"❌ {message}")
        return False
    print(f"✅ {message}")
    return True


def test_stop_aggregation_review_stats():
    """Test that stop-level aggregation includes review statistics."""
    print("Testing stop-level aggregation review statistics...")
    
    # Create test segments with different review conditions
    segments = [
        # Normal segment (no review needed)
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            asr_confidence=0.9,
            translation_confidence=0.9,
            sentiment_label='positive',
            churn_risk='low',
            role_confidence=0.4
        ),
        # ASR low confidence (needs review)
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            asr_confidence=0.6,  # < 0.65
            translation_confidence=0.9,
            sentiment_label='neutral',
            churn_risk='low',
            role_confidence=0.4
        ),
        # Translation low confidence (needs review)
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            asr_confidence=0.9,
            translation_confidence=0.5,  # < 0.65
            sentiment_label='neutral',
            churn_risk='low',
            role_confidence=0.4
        ),
        # High risk negative (needs review)
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            asr_confidence=0.9,
            translation_confidence=0.9,
            sentiment_label='negative',
            churn_risk='high',
            role_confidence=0.6  # >= 0.5
        )
    ]
    
    # Create aggregator
    config = {'placeholders': {}}
    aggregator = Aggregator(config)
    
    # Test stop-level aggregation
    stop_aggregates = aggregator.aggregate_by_stop(segments)
    
    if 'STOP45' not in stop_aggregates:
        return expect(False, "Stop aggregate not found")
    
    stop_agg = stop_aggregates['STOP45']
    
    # Check that review_statistics field exists
    if 'review_statistics' not in stop_agg:
        return expect(False, "review_statistics field missing from stop aggregate")
    
    review_stats = stop_agg['review_statistics']
    
    # Verify expected values
    success = True
    success &= expect(review_stats['total_segments'] == 4, f"Total segments: {review_stats['total_segments']}")
    success &= expect(review_stats['needs_review'] == 3, f"Needs review: {review_stats['needs_review']}")
    success &= expect(review_stats['review_percentage'] == 75.0, f"Review percentage: {review_stats['review_percentage']}")
    
    # Check review reasons
    reasons = review_stats['review_reasons']
    success &= expect(reasons['asr_low_confidence'] == 1, f"ASR low confidence: {reasons['asr_low_confidence']}")
    success &= expect(reasons['translation_low_confidence'] == 1, f"Translation low confidence: {reasons['translation_low_confidence']}")
    success &= expect(reasons['high_risk_negative'] == 1, f"High risk negative: {reasons['high_risk_negative']}")
    
    return success


def test_day_aggregation_review_stats():
    """Test that day-level aggregation includes review statistics."""
    print("\nTesting day-level aggregation review statistics...")
    
    # Create test segments across multiple stops
    segments = [
        # Stop 1 - normal segment
        Segment(
            seller_id="S123",
            stop_id="STOP45",
            timestamp="2025-08-19T10:00:00Z",
            asr_confidence=0.9,
            translation_confidence=0.9,
            sentiment_label='positive',
            churn_risk='low',
            role_confidence=0.4
        ),
        # Stop 2 - needs review
        Segment(
            seller_id="S123",
            stop_id="STOP46",
            timestamp="2025-08-19T11:00:00Z",
            asr_confidence=0.6,
            translation_confidence=0.9,
            sentiment_label='neutral',
            churn_risk='low',
            role_confidence=0.4
        )
    ]
    
    # Create aggregator
    config = {'placeholders': {}}
    aggregator = Aggregator(config)
    
    # Test day-level aggregation
    day_aggregates = aggregator.aggregate_by_day(segments)
    
    if 'S123' not in day_aggregates:
        return expect(False, "Day aggregate not found")
    
    day_agg = day_aggregates['S123']
    
    # Check that review_statistics field exists
    if 'review_statistics' not in day_agg:
        return expect(False, "review_statistics field missing from day aggregate")
    
    review_stats = day_agg['review_statistics']
    
    # Verify expected values
    success = True
    success &= expect(review_stats['total_segments'] == 2, f"Total segments: {review_stats['total_segments']}")
    success &= expect(review_stats['needs_review'] == 1, f"Needs review: {review_stats['needs_review']}")
    success &= expect(review_stats['review_percentage'] == 50.0, f"Review percentage: {review_stats['review_percentage']}")
    
    # Check review reasons
    reasons = review_stats['review_reasons']
    success &= expect(reasons['asr_low_confidence'] == 1, f"ASR low confidence: {reasons['asr_low_confidence']}")
    success &= expect(reasons['translation_low_confidence'] == 0, f"Translation low confidence: {reasons['translation_low_confidence']}")
    success &= expect(reasons['high_risk_negative'] == 0, f"High risk negative: {reasons['high_risk_negative']}")
    
    return success


def main():
    tests = [
        test_stop_aggregation_review_stats,
        test_day_aggregation_review_stats,
    ]
    passed = 0
    for t in tests:
        if t():
            passed += 1
    print(f"\nAggregation review stats tests: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
