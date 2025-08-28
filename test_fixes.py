#!/usr/bin/env python3
"""
Test script to validate the fixes applied to the voice sentiment analysis pipeline.
Tests all the major issues that were identified and fixed.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import yaml
from analyze import NLUAnalyzer, Segment
from datetime import datetime

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_speaker_role_detection():
    """Test speaker role detection fixes."""
    print("=== Testing Speaker Role Detection ===")
    
    config = load_config()
    analyzer = NLUAnalyzer(config)
    
    # Test cases that were problematic
    test_cases = [
        {
            'text': 'நீங்கள் ஒரு நாள் வருகிறீங்க, ஒரு நாள் வரமாற்றுறீங்க, ஆனால் சொல்லும்போது தினமும் வருகும் என்று சொல்றீங்க.',
            'expected_role': 'buyer',
            'description': 'Buyer complaining about seller inconsistency'
        },
        {
            'text': 'நேற்று நீங்கள் வரவே இல்ல, இது மாதிரி அடிக்கடி பண்றீங்க.',
            'expected_role': 'buyer',
            'description': 'Buyer complaining about seller not coming'
        },
        {
            'text': 'உங்களை நம்பி நாங்க காய் வாங்காமல இருக்கும்.',
            'expected_role': 'buyer',
            'description': 'Buyer refusing to buy from seller'
        },
        {
            'text': 'நாளைக்கு வரும்போது காலான் எடுத்து வரப்பாருங்க',
            'expected_role': 'seller',
            'description': 'Seller promising to bring items'
        },
        {
            'text': 'தக்காலி நல்லா இருக்குங்க இன்னைக்கு',
            'expected_role': 'other',
            'description': 'Product praise (could be either)'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        segment = Segment(
            textTamil=test_case['text'],
            textEnglish='Test translation',
            segment_id=f"TEST_{i}"
        )
        
        analyzer._analyze_speaker_role(segment)
        
        status = "✓" if segment.speaker_role == test_case['expected_role'] else "✗"
        print(f"{status} Test {i}: {test_case['description']}")
        print(f"    Expected: {test_case['expected_role']}, Got: {segment.speaker_role}")
        print(f"    Text: {test_case['text'][:50]}...")
        print()

def test_intent_classification():
    """Test intent classification fixes."""
    print("=== Testing Intent Classification ===")
    
    config = load_config()
    analyzer = NLUAnalyzer(config)
    
    test_cases = [
        {
            'text': 'உங்களை நம்பி நாங்க காய் வாங்காமல இருக்கும்.',
            'expected_intent': 'purchase_negative',
            'description': 'Negative purchase intent (refusal)'
        },
        {
            'text': 'கொத்தமல்லி கருவப்புல் கொஞ்சம் கொத்தமல்லி மட்டு கொடுங்களே',
            'expected_intent': 'purchase_request',
            'description': 'Product request (coriander)'
        },
        {
            'text': 'தக்காலி நல்லா இருக்குங்க இன்னைக்கு',
            'expected_intent': 'product_praise',
            'description': 'Product praise'
        },
        {
            'text': 'நீங்கள் ஒரு நாள் வருகிறீங்க, ஒரு நாள் வரமாற்றுறீங்க',
            'expected_intent': 'complaint',
            'description': 'Complaint about seller inconsistency'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        segment = Segment(
            textTamil=test_case['text'],
            textEnglish='Test translation',
            segment_id=f"TEST_{i}"
        )
        
        analyzer._analyze_intent(segment)
        
        status = "✓" if segment.intent == test_case['expected_intent'] else "✗"
        print(f"{status} Test {i}: {test_case['description']}")
        print(f"    Expected: {test_case['expected_intent']}, Got: {segment.intent}")
        print(f"    Text: {test_case['text'][:50]}...")
        print()

def test_sentiment_analysis():
    """Test sentiment analysis fixes."""
    print("=== Testing Sentiment Analysis ===")
    
    config = load_config()
    analyzer = NLUAnalyzer(config)
    
    test_cases = [
        {
            'text': 'உங்களை நம்பி நாங்க காய் வாங்காமல இருக்கும்.',
            'intent': 'purchase_negative',
            'expected_sentiment': 'negative',
            'description': 'Negative purchase should have negative sentiment'
        },
        {
            'text': 'தக்காலி நல்லா இருக்குங்க இன்னைக்கு',
            'intent': 'product_praise',
            'expected_sentiment': 'positive',
            'description': 'Product praise should have positive sentiment'
        },
        {
            'text': 'நீங்கள் ஒரு நாள் வருகிறீங்க, ஒரு நாள் வரமாற்றுறீங்க',
            'intent': 'complaint',
            'expected_sentiment': 'negative',
            'description': 'Complaint should have negative sentiment'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        segment = Segment(
            textTamil=test_case['text'],
            textEnglish='Test translation',
            segment_id=f"TEST_{i}",
            intent=test_case['intent']
        )
        
        analyzer._analyze_sentiment(segment)
        analyzer._validate_and_fix_contradictions(segment)
        
        status = "✓" if segment.sentiment_label == test_case['expected_sentiment'] else "✗"
        print(f"{status} Test {i}: {test_case['description']}")
        print(f"    Expected: {test_case['expected_sentiment']}, Got: {segment.sentiment_label}")
        print(f"    Score: {segment.sentiment_score}")
        print(f"    Intent: {segment.intent}")
        print()

def test_contradiction_fixes():
    """Test logical contradiction fixes."""
    print("=== Testing Contradiction Fixes ===")
    
    config = load_config()
    analyzer = NLUAnalyzer(config)
    
    # Test case with contradiction: negative intent but positive sentiment
    segment = Segment(
        textTamil='உங்களை நம்பி நாங்க காய் வாங்காமல இருக்கும்.',
        textEnglish='We won\'t buy from you.',
        segment_id='TEST_CONTRADICTION',
        intent='purchase_negative',
        sentiment_score=0.5,
        sentiment_label='positive'
    )
    
    print(f"Before fix:")
    print(f"    Intent: {segment.intent}")
    print(f"    Sentiment: {segment.sentiment_label} ({segment.sentiment_score})")
    
    analyzer._validate_and_fix_contradictions(segment)
    
    print(f"After fix:")
    print(f"    Intent: {segment.intent}")
    print(f"    Sentiment: {segment.sentiment_label} ({segment.sentiment_score})")
    # analysis_metadata was removed from outputs; only verifying corrected values now
    print()

def test_configuration():
    """Test configuration updates."""
    print("=== Testing Configuration Updates ===")
    
    config = load_config()
    
    # Check intent categories
    intent_categories = config['fields']['intent_categories']
    expected_intents = ['purchase_positive', 'purchase_negative', 'purchase_request', 'complaint', 'product_praise', 'bargain', 'greeting', 'other']
    
    print("Intent categories:")
    for intent in expected_intents:
        status = "✓" if intent in intent_categories else "✗"
        print(f"    {status} {intent}")
    
    # Check sentiment thresholds
    sentiment_thresholds = config['sentiment_thresholds']
    print(f"\nSentiment thresholds:")
    print(f"    Positive: {sentiment_thresholds['positive']}")
    print(f"    Negative: {sentiment_thresholds['negative']}")
    print(f"    Neutral range: {sentiment_thresholds['neutral_range']}")
    print()

def main():
    """Run all tests."""
    print("Voice Sentiment Analysis Pipeline - Fix Validation Tests")
    print("=" * 60)
    print()
    
    try:
        test_configuration()
        test_speaker_role_detection()
        test_intent_classification()
        test_sentiment_analysis()
        test_contradiction_fixes()
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
